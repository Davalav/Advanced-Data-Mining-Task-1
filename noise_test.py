# ------------------ Noise robustness -----------------
# Lägger NSTDB-brus på EKG-signalen och testar FP32- och INT8-modellen

import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.ao.quantization as quantization
import torch.nn as nn
import wfdb
from ecgdetectors import Detectors
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

import lib

# Sätt till True först när allt annat är låst, DS2 ska bara köras en gång
RUN_TEST = False

SEEDS = [42, 1, 2]
FILTERS = [16, 32, 64]
NOISE_TYPES = ["bw", "em", "ma"]
SNR_DB = [12, 6, 0]
PEAK_TOLERANCE = 0.15   # sekunder, en detekterad topp så nära slaget räknas som slaget självt
window_size = 100
offset = 0

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
NOISE_DIR = os.path.join(HERE, "..", "mit-bih-noise-stress-test-database-1.0.0") + os.sep

split = "test" if RUN_TEST else "val"
records = lib.test_records if RUN_TEST else lib.val_records
RESULTS_FILE = os.path.join(HERE, "results", f"noise_results_{split}.csv")

torch.backends.quantized.engine = "onednn"
half_window = window_size // 2


def kept_beats(annotation, signal_length):
    # Samma slag som MITBIHDataset använder
    starts = annotation.sample - half_window + offset
    ends = annotation.sample + half_window + offset
    keep = np.isin(annotation.symbol, list(lib.AAMI_MAPPING)) & (starts >= 0) & (ends < signal_length)
    labels = [lib.AAMI_MAPPING[symbol] for symbol in np.array(annotation.symbol)[keep]]
    return annotation.sample[keep], np.array(labels)


def annotated_peaks(annotation):
    return annotation.sample[np.isin(annotation.symbol, lib.RR_BEAT_SYMBOLS)]


def rr_intervals(beat_samples, peaks, fs):
    # Tid mellan slagets egen topp och närmaste topp före och efter, i sekunder
    tolerance = PEAK_TOLERANCE * fs
    peaks = np.sort(np.asarray(peaks, dtype=float))
    padded = np.concatenate([[np.nan], peaks, [np.nan]])

    # Slagets egen topp är närmaste topp inom toleransen, annars används annoteringen.
    # Detektorn markerar toppen lite senare än annoteringen, så RR mäts mellan toppar
    i = np.searchsorted(peaks, beat_samples)
    distance_left = np.nan_to_num(beat_samples - padded[i], nan=np.inf)
    distance_right = np.nan_to_num(padded[i + 1] - beat_samples, nan=np.inf)
    nearest = np.where(distance_left <= distance_right, padded[i], padded[i + 1])
    own = np.where(np.minimum(distance_left, distance_right) <= tolerance, nearest, beat_samples)

    before = np.searchsorted(peaks, own - tolerance, side="left")
    after = np.searchsorted(peaks, own + tolerance, side="right") + 1
    pre_rr = (own - padded[before]) / fs
    post_rr = (padded[after] - own) / fs

    # Saknas en granne används den andra, som i lib.py
    pre_rr = np.where(np.isnan(pre_rr), post_rr, pre_rr)
    post_rr = np.where(np.isnan(post_rr), pre_rr, post_rr)
    return np.stack([pre_rr, post_rr], axis=1)


def found_by_detector(beat_samples, peaks, fs):
    tolerance = PEAK_TOLERANCE * fs
    peaks = np.sort(np.asarray(peaks, dtype=float))
    return (np.searchsorted(peaks, beat_samples + tolerance, side="right")
            - np.searchsorted(peaks, beat_samples - tolerance, side="left")) > 0


def add_noise(signal, noise, snr_db):
    # Skalar bruset så att signal/brus-förhållandet blir snr_db
    noise = noise[:len(signal)]
    scale = np.sqrt(np.var(signal) / (np.var(noise) * 10 ** (snr_db / 10)))
    return signal + scale * noise


def beat_windows(signal, beat_samples, fs):
    # Samma filter och z-normalisering per fönster som MITBIHDataset
    filtered = lib.butter_bandpass_filter(signal, lib.lpf_cutoff, lib.hpf_cutoff, fs)
    filtered = lib.notch_filter(filtered, 50, fs)
    windows = np.stack([filtered[s - half_window + offset:s + half_window + offset] for s in beat_samples])
    std = windows.std(axis=1, keepdims=True)
    windows = (windows - windows.mean(axis=1, keepdims=True)) / np.where(std > 0, std, 1)
    return windows.astype(np.float32)


# RR normaliseras med träningsdatans medelvärde och standardavvikelse, som i cachen
train_rr = []
for record_name in lib.train_records:
    annotation = wfdb.rdann(lib.directory + record_name, 'atr')
    header = wfdb.rdheader(lib.directory + record_name)
    beat_samples, _ = kept_beats(annotation, header.sig_len)
    train_rr.append(rr_intervals(beat_samples, annotated_peaks(annotation), header.fs))
train_rr = np.concatenate(train_rr)
rr_mean, rr_std = train_rr.mean(axis=0), train_rr.std(axis=0)

clean_records = {}
for record_name in records:
    record = wfdb.rdrecord(lib.directory + record_name)
    annotation = wfdb.rdann(lib.directory + record_name, 'atr')
    clean_records[record_name] = (record.p_signal[:, 0], annotation, record.fs)

noise_signals = {name: wfdb.rdrecord(NOISE_DIR + name).p_signal[:, 0] for name in NOISE_TYPES}


def build_inputs(noise_type, snr_db, rr_source):
    X, y, found = [], [], []
    for signal, annotation, fs in clean_records.values():
        if noise_type != "clean":
            signal = add_noise(signal, noise_signals[noise_type], snr_db)
        beat_samples, labels = kept_beats(annotation, len(signal))

        if rr_source == "detector":
            peaks = Detectors(fs).pan_tompkins_detector(signal)
        else:
            peaks = annotated_peaks(annotation)
        found.append(found_by_detector(beat_samples, peaks, fs))

        rr = (rr_intervals(beat_samples, peaks, fs) - rr_mean) / rr_std
        rr = np.nan_to_num(rr)   # inga toppar alls: medelvärdet används
        rr_channels = np.repeat(rr[:, :, None], window_size, axis=2)

        windows = beat_windows(signal, beat_samples, fs)
        X.append(np.concatenate([windows[:, None, :], rr_channels], axis=1))
        y.append(labels)

    return torch.tensor(np.concatenate(X), dtype=torch.float32), np.concatenate(y), np.concatenate(found).mean()


def build_model():
    layers = []
    in_channels = 3
    length = window_size
    for out_channels in FILTERS:
        layers += [nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2)]
        in_channels = out_channels
        length //= 2
    return nn.Sequential(*layers, nn.Flatten(), nn.Linear(in_channels * length, 4))


def load_fp32(path):
    model = build_model()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


def load_int8(path):
    # Samma steg som i quantize.py, men skalorna läses från filen i stället för att kalibreras
    conv_relu_pairs = [[str(i), str(i + 1)] for i in range(0, 3 * len(FILTERS), 3)]
    model = quantization.QuantWrapper(quantization.fuse_modules(build_model(), conv_relu_pairs)).eval()
    model.qconfig = quantization.get_default_qconfig("onednn")
    quantization.prepare(model, inplace=True)
    with torch.no_grad():
        model(torch.zeros(1, 3, window_size))
    model = quantization.convert(model)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    return model


def predict(model, X):
    with torch.no_grad():
        return torch.cat([model(batch).argmax(dim=1) for batch in X.split(256)]).numpy()


def metrics(y, y_pred):
    result = {
        "accuracy": accuracy_score(y, y_pred),
        "f1_macro": f1_score(y, y_pred, average='macro'),
    }
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, labels=[0, 1, 2, 3], zero_division=0)
    for i, cls in enumerate(["N", "S", "V", "F"]):
        result[f"precision_{cls}"] = precision[i]
        result[f"recall_{cls}"] = recall[i]
        result[f"f1_{cls}"] = f1[i]
    return result


filters_name = "-".join(map(str, FILTERS))
models = []
for seed in SEEDS:
    fp32_file, = glob.glob(os.path.join(MODELS_DIR, f"SmallCNN-valF1-*-seed{seed}-lr0.02-cw0-sampler1-rr1-filters{filters_name}.pth"))
    int8_file, = glob.glob(os.path.join(MODELS_DIR, f"SmallCNN-INT8-valF1-*-seed{seed}-filters{filters_name}.pth"))
    models += [("FP32", seed, load_fp32(fp32_file)), ("INT8", seed, load_int8(int8_file))]

# Ren signal med annoteringarnas RR är samma som de tidigare resultaten,
# alla andra rader räknar RR från detektorn som på en riktig enhet
conditions = [("clean", None, "annotations"), ("clean", None, "detector")]
conditions += [(noise_type, snr_db, "detector") for noise_type in NOISE_TYPES for snr_db in SNR_DB]

results = []
for noise_type, snr_db, rr_source in conditions:
    X, y, beats_found = build_inputs(noise_type, snr_db, rr_source)
    for name, seed, model in models:
        results.append({"split": split, "noise": noise_type, "snr_db": snr_db, "rr_source": rr_source,
                        "format": name, "seed": seed, "beats_found": beats_found, **metrics(y, predict(model, X))})

    f1_mean = pd.DataFrame(results[-len(models):]).groupby("format")["f1_macro"].mean()
    snr_text = "" if snr_db is None else str(snr_db)
    print(f"{noise_type:5s} {snr_text:>2s} dB  RR från {rr_source:11s}  hittade slag {beats_found * 100:6.2f} %"
          f"  F1 FP32 {f1_mean['FP32'] * 100:5.2f} %  INT8 {f1_mean['INT8'] * 100:5.2f} %", flush=True)

results = pd.DataFrame(results)

# Medelvärde över seeds läggs sist i samma CSV
group = ["noise", "snr_db", "rr_source", "format"]
grouped = results.drop(columns=["seed", "split"]).groupby(group, sort=False, dropna=False)
means = grouped.mean().reset_index()
means["f1_std"] = grouped["f1_macro"].std().values
means["seed"] = "mean"
means["split"] = split

results = pd.concat([results, means], ignore_index=True).round(4)
results.to_csv(RESULTS_FILE, index=False)
print(f"\nSparade resultat till {RESULTS_FILE}")
