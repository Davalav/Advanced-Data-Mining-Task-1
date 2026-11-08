# ------------------ Error analysis -----------------
# Väljer typiska fel och ritar EKG-kurvorna som modellen såg

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wfdb
from ecgdetectors import Detectors

import dataset
import model as cnn
from config import CONFIG

# Sätt till True först när allt annat är låst, DS2 ska bara köras en gång
RUN_TEST = True

SEED = CONFIG["deployed_seed"]                       # vilken av de tre INT8-modellerna som analyseras
FILTERS = CONFIG["filters"]
NOISE_TYPE, NOISE_SNR_DB = "em", 0
EXAMPLES_PER_TYPE = 3
CONTEXT_SECONDS = 1.5           # hur mycket signal som visas före och efter slaget
window_size = CONFIG["window_size"]
offset = CONFIG["offset"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOISE_DIR = dataset.noise_directory
split = "test" if RUN_TEST else "val"
records = dataset.test_records if RUN_TEST else dataset.val_records
OUT_DIR = os.path.join(ROOT, "results", "error_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

CLASSES = ["N", "S", "V", "F"]
INK, SECONDARY, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
TRUE_COLOR, PREDICTED_COLOR = "#2a78d6", "#eb6834"

rr_mean, rr_std = dataset.train_rr_stats(window_size, offset)
noise = wfdb.rdrecord(NOISE_DIR + NOISE_TYPE).p_signal[:, 0]

int8_file, = glob.glob(os.path.join(ROOT, "models", f"SmallCNN-INT8-valF1-*-seed{SEED}-filters{'-'.join(map(str, FILTERS))}.pth"))
model = cnn.load_cnn_int8(int8_file, FILTERS, window_size)


def load_beats(noisy):
    # Ren signal använder annoteringarnas RR, brusig signal RR från detektorn, som i noise_test.py
    signals, rows, windows, rr = {}, [], [], []
    for record_name in records:
        record = wfdb.rdrecord(dataset.directory + record_name)
        annotation = wfdb.rdann(dataset.directory + record_name, 'atr')
        fs = record.fs
        signal = record.p_signal[:, 0]
        if noisy:
            signal = dataset.add_noise(signal, noise, NOISE_SNR_DB)
            peaks = np.sort(Detectors(fs).pan_tompkins_detector(signal))
        else:
            peaks = dataset.annotated_peaks(annotation)

        beat_samples, labels = dataset.kept_beats(annotation, len(signal), window_size, offset)
        filtered = dataset.filter_signal(signal, fs)
        signals[record_name] = (filtered, annotation, peaks, fs)
        rows += [(record_name, sample, label) for sample, label in zip(beat_samples, labels)]
        windows.append(dataset.beat_windows(filtered, beat_samples, window_size, offset))
        rr.append(dataset.rr_intervals(beat_samples, peaks, fs))

    beats = pd.DataFrame(rows, columns=["record", "sample", "label"])
    rr = np.concatenate(rr)
    beats["pre_rr"], beats["post_rr"] = rr[:, 0], rr[:, 1]
    windows = np.concatenate(windows)

    rr_channels = np.repeat(np.nan_to_num((rr - rr_mean) / rr_std)[:, :, None], window_size, axis=2)
    X = torch.tensor(np.concatenate([windows[:, None, :], rr_channels], axis=1), dtype=torch.float32)
    with torch.no_grad():
        probabilities = torch.cat([torch.softmax(model(batch), dim=1) for batch in X.split(256)]).numpy()
    beats["predicted"] = probabilities.argmax(axis=1)
    beats["confidence"] = probabilities.max(axis=1)
    return beats, windows, signals


clean, clean_windows, clean_signals = load_beats(noisy=False)
noisy, noisy_windows, noisy_signals = load_beats(noisy=True)

# Medelslag per klass på ren signal, att jämföra varje fel med
templates = [clean_windows[clean["label"] == c].mean(axis=0) for c in range(len(CLASSES))]

clean_right = clean["label"] == clean["predicted"]
noise_name = f"{NOISE_TYPE} {NOISE_SNR_DB} dB"
error_types = [
    ("N_as_S", "N classified as S", clean, clean_windows, clean_signals,
     (clean["label"] == 0) & (clean["predicted"] == 1), clean["label"] == 0),
    ("S_as_N", "S classified as N", clean, clean_windows, clean_signals,
     (clean["label"] == 1) & (clean["predicted"] == 0), clean["label"] == 1),
    ("V_wrong", "V classified incorrectly", clean, clean_windows, clean_signals,
     (clean["label"] == 2) & (clean["predicted"] != 2), clean["label"] == 2),
    ("N_as_F", "N classified as F", clean, clean_windows, clean_signals,
     (clean["label"] == 0) & (clean["predicted"] == 3), clean["label"] == 0),
    (f"noise_{NOISE_TYPE}{NOISE_SNR_DB}dB", f"Correct on clean signal, wrong at {noise_name}", noisy, noisy_windows, noisy_signals,
     clean_right & (noisy["label"] != noisy["predicted"]), clean_right),
]


def plot_examples(file_name, title, beats, windows, signals, chosen):
    fig, axes = plt.subplots(len(chosen), 2, figsize=(12, 2.8 * len(chosen)), squeeze=False,
                             gridspec_kw={"width_ratios": [2.2, 1]}, facecolor=SURFACE)
    half_window = window_size // 2

    for row, i in enumerate(chosen):
        beat = beats.loc[i]
        true_class, predicted_class = CLASSES[beat["label"]], CLASSES[beat["predicted"]]
        filtered, annotation, peaks, fs = signals[beat["record"]]
        sample = beat["sample"]

        # Vänster: några sekunder runt slaget, så att rytmen syns
        ax = axes[row, 0]
        start = max(0, sample - int(CONTEXT_SECONDS * fs))
        end = min(len(filtered), sample + int(CONTEXT_SECONDS * fs))
        time = (np.arange(start, end) - sample) / fs
        if beats is noisy:
            ax.plot(time, clean_signals[beat["record"]][0][start:end], color=MUTED, linewidth=1, label="Clean signal")
        ax.plot(time, filtered[start:end], color=INK, linewidth=1, label="Signal the model saw")
        ax.axvspan((-half_window + offset) / fs, (half_window + offset) / fs, color=TRUE_COLOR, alpha=0.12, linewidth=0)

        top = filtered[start:end].max()
        for other, symbol in zip(annotation.sample, annotation.symbol):
            if start <= other < end and symbol in dataset.RR_BEAT_SYMBOLS:
                ax.text((other - sample) / fs, top, symbol, ha="center", va="bottom", color=SECONDARY, fontsize=8)
        if beats is noisy:
            shown = peaks[(peaks >= start) & (peaks < end)]
            ax.plot((shown - sample) / fs, np.full(len(shown), filtered[start:end].min()), "^", color=PREDICTED_COLOR,
                    markersize=6, label="Detected peaks")

        ax.set_title(f"Record {beat['record']} at {sample / fs:.1f} s: true {true_class}, predicted {predicted_class} "
                     f"({beat['confidence'] * 100:.0f} %), RR before {beat['pre_rr']:.2f} s, after {beat['post_rr']:.2f} s",
                     loc="left", fontsize=9, color=INK)

        # Höger: fönstret modellen klassificerade, jämfört med medelslagen
        ax = axes[row, 1]
        window_ms = (np.arange(window_size) - half_window + offset) / fs * 1000
        ax.plot(window_ms, templates[beat["label"]], color=TRUE_COLOR, linewidth=2, label=f"Average {true_class}")
        ax.plot(window_ms, templates[beat["predicted"]], color=PREDICTED_COLOR, linewidth=2, label=f"Average {predicted_class}")
        ax.plot(window_ms, windows[i], color=INK, linewidth=1.5, label="This beat")
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=SECONDARY, labelsize=8)
        ax.grid(color="#e5e4e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color(MUTED)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="upper right", ncol=3)
    axes[-1, 0].set_xlabel("Time from beat (s), shaded area is the window the model sees", color=SECONDARY, fontsize=9)
    axes[-1, 1].set_xlabel("Time from beat (ms)", color=SECONDARY, fontsize=9)

    fig.suptitle(f"{title} ({split} data, INT8 model seed {SEED})", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, file_name), dpi=130, facecolor=SURFACE)
    plt.close(fig)


rng = np.random.default_rng(0)
counts, examples = [], []
for key, title, beats, windows, signals, errors, total in error_types:
    error_index = np.flatnonzero(errors)
    counts.append({"error": title, "errors": len(error_index), "out_of": int(total.sum()),
                   "share": len(error_index) / max(int(total.sum()), 1)})
    print(f"{title}: {len(error_index)} av {int(total.sum())}")
    if len(error_index) == 0:
        continue

    # Slumpvis valda fel, så att exemplen är typiska och inte bara de mest extrema
    chosen = np.sort(rng.choice(error_index, size=min(EXAMPLES_PER_TYPE, len(error_index)), replace=False))
    plot_examples(f"{split}_{key}.png", title, beats, windows, signals, chosen)
    for i in chosen:
        beat = beats.loc[i]
        examples.append({"error": title, "record": beat["record"], "time_s": round(beat["sample"] / 360, 2),
                         "sample": beat["sample"], "true": CLASSES[beat["label"]], "predicted": CLASSES[beat["predicted"]],
                         "confidence": beat["confidence"], "pre_rr": beat["pre_rr"], "post_rr": beat["post_rr"]})

pd.DataFrame(counts).round(4).to_csv(os.path.join(OUT_DIR, f"{split}_error_counts.csv"), index=False)
pd.DataFrame(examples).round(3).to_csv(os.path.join(OUT_DIR, f"{split}_examples.csv"), index=False)
print(f"\nSparade bilder och CSV-filer i {OUT_DIR}")
