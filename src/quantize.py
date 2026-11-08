# ------------------ TASK 4.5 FP32 vs INT8 -----------------
# Gör om den valda CNN + RR-modellen till INT8 och jämför på valideringsdatan

import copy
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.ao.quantization as quantization
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

import dataset
from config import CONFIG
from prepare_data import CACHE_FILE

SEEDS = CONFIG["seeds"]
FILTERS = CONFIG["filters"]
CALIBRATION_BEATS = CONFIG["calibration_beats"]
LATENCY_BEATS = 1000
window_size = CONFIG["window_size"]
offset = CONFIG["offset"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_FILE = os.path.join(ROOT, "results", "quantization_metrics.csv")

torch.backends.quantized.engine = "onednn"
torch.set_num_threads(1)

train_dataset = dataset.MITBIHDataset(record_list=dataset.train_records, window_size=window_size, offset=offset)
val_dataset = dataset.MITBIHDataset(record_list=dataset.val_records, window_size=window_size, offset=offset)
y_val = val_dataset.labels

cache = np.load(CACHE_FILE)
feature_names = list(cache["feature_names"])
rr_idx = [feature_names.index("pre_rr"), feature_names.index("post_rr")]


def add_rr_channels(X, features):
    rr = torch.tensor(features[:, rr_idx], dtype=torch.float32)
    rr = rr.unsqueeze(2).repeat(1, 1, X.shape[2])
    return torch.cat([X, rr], dim=1)


X_train = add_rr_channels(torch.tensor(train_dataset.beats).unsqueeze(1), cache["X_train"])
X_val = add_rr_channels(torch.tensor(val_dataset.beats).unsqueeze(1), cache["X_val"])


def build_model():
    layers = []
    in_channels = X_train.shape[1]
    length = X_train.shape[2]
    for out_channels in FILTERS:
        layers += [nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2)]
        in_channels = out_channels
        length //= 2
    return nn.Sequential(*layers, nn.Flatten(), nn.Linear(in_channels * length, 4))


def to_int8(model, seed):
    # Conv och ReLU slås ihop till ett steg, så att de kvantiseras tillsammans
    conv_relu_pairs = [[str(i), str(i + 1)] for i in range(0, 3 * len(FILTERS), 3)]
    fused = quantization.fuse_modules(copy.deepcopy(model), conv_relu_pairs)

    int8_model = quantization.QuantWrapper(fused).eval()
    int8_model.qconfig = quantization.get_default_qconfig("onednn")
    quantization.prepare(int8_model, inplace=True)

    # Kalibrering: mäter vilka talområden lagren använder, vikterna ändras inte
    generator = torch.Generator().manual_seed(seed)
    beats = torch.randperm(len(X_train), generator=generator)[:CALIBRATION_BEATS]
    with torch.no_grad():
        for batch in X_train[beats].split(256):
            int8_model(batch)

    return quantization.convert(int8_model)


def predict(model):
    with torch.no_grad():
        return torch.cat([model(batch).argmax(dim=1) for batch in X_val.split(256)]).numpy()


def latency_ms(model):
    times = []
    with torch.no_grad():
        for beat in X_val[:LATENCY_BEATS]:
            start_time = time.perf_counter()
            model(beat.unsqueeze(0))
            times.append(time.perf_counter() - start_time)
    return np.median(times) * 1000


def evaluate(model, y_pred, fp32_pred, name, seed, bytes_per_weight, model_file):
    print(f"\n--- {name}, seed {seed} ---")
    print(classification_report(y_val, y_pred, target_names=dataset.CLASS_NAMES, digits=4, zero_division=0))
    print(confusion_matrix(y_val, y_pred))

    n_parameters = sum(p.numel() for p in build_model().parameters())
    result = {
        "format": name,
        "seed": seed,
        "split": "val",
        "n_parameters": n_parameters,
        "weights_kb": n_parameters * bytes_per_weight / 1000,
        "file_kb": os.path.getsize(model_file) / 1000,
        "pc_latency_ms": latency_ms(model),   # mätt på datorns processor, inte på MAX78002
        "same_as_fp32": (y_pred == fp32_pred).mean(),
        "accuracy": accuracy_score(y_val, y_pred),
        "f1_macro": f1_score(y_val, y_pred, average='macro'),
    }
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, labels=[0, 1, 2, 3], zero_division=0)
    for i, cls in enumerate(["N", "S", "V", "F"]):
        result[f"precision_{cls}"] = precision[i]
        result[f"recall_{cls}"] = recall[i]
        result[f"f1_{cls}"] = f1[i]
    return result


filters_name = "-".join(map(str, FILTERS))
results = []

for seed in SEEDS:
    pattern = f"SmallCNN-valF1-*-seed{seed}-lr0.02-cw0-sampler1-rr1-filters{filters_name}.pth"
    fp32_file, = glob.glob(os.path.join(MODELS_DIR, pattern))

    fp32_model = build_model()
    fp32_model.load_state_dict(torch.load(fp32_file, map_location="cpu"))
    fp32_model.eval()
    fp32_pred = predict(fp32_model)
    results.append(evaluate(fp32_model, fp32_pred, fp32_pred, "FP32", seed, 4, fp32_file))

    int8_model = to_int8(fp32_model, seed)
    int8_pred = predict(int8_model)
    int8_f1 = f1_score(y_val, int8_pred, average='macro')
    int8_file = os.path.join(MODELS_DIR, f"SmallCNN-INT8-valF1-{int8_f1:.4f}-seed{seed}-filters{filters_name}.pth")
    torch.save(int8_model.state_dict(), int8_file)
    results.append(evaluate(int8_model, int8_pred, fp32_pred, "INT8", seed, 1, int8_file))

results = pd.DataFrame(results)

# Medelvärde per format läggs sist i samma CSV
means = results.drop(columns=["seed", "split"]).groupby("format", sort=False).mean().reset_index()
means["seed"] = "mean"
means["split"] = "val"
means["f1_std"] = results.groupby("format", sort=False)["f1_macro"].std().values

results = pd.concat([results, means], ignore_index=True).round(4)
results.to_csv(RESULTS_FILE, index=False)
print(f"\nSparade resultat till {RESULTS_FILE}")
print(means[["format", "f1_macro", "f1_std", "accuracy", "weights_kb", "file_kb", "pc_latency_ms", "same_as_fp32"]].round(4))
