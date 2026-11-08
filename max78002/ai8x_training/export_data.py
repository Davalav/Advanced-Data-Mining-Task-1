# ------------------ Export för Analog Devices ai8x-training -----------------
# Sparar samma fönster och RR som vår CNN använder, skalade till 0-1 som ai8x vill ha.
# Körs i projektets vanliga Python-miljö, inte i ai8x-miljön.

import os
import sys

import numpy as np
import wfdb

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
sys.path.insert(0, REPO)
import lib  # noqa: E402

# Sätt till True först när allt annat är låst, DS2 ska bara köras en gång
RUN_TEST = False

window_size = 100
offset = 0
INPUT_RANGE = 4.0   # z-värden mellan -4 och +4 blir 0-1, större värden klipps
OUT_FILE = os.path.join(REPO, "..", "ai8x_data", "ECG_RR", "ecg_rr.npz")

rr_mean, rr_std = lib.train_rr_stats(window_size, offset)


def build(records):
    X, y = [], []
    for record_name in records:
        record = wfdb.rdrecord(lib.directory + record_name)
        annotation = wfdb.rdann(lib.directory + record_name, 'atr')
        signal = record.p_signal[:, 0]
        beat_samples, labels = lib.kept_beats(annotation, len(signal), window_size, offset)
        rr = (lib.rr_intervals(beat_samples, lib.annotated_peaks(annotation), record.fs) - rr_mean) / rr_std
        windows = lib.beat_windows(lib.filter_signal(signal, record.fs), beat_samples, window_size, offset)
        X.append(np.concatenate([windows[:, None, :], np.repeat(rr[:, :, None], window_size, axis=2)], axis=1))
        y.append(labels)
    X = np.concatenate(X) / (2 * INPUT_RANGE) + 0.5
    print(f"{len(X)} slag, {np.mean((X < 0) | (X > 1)) * 100:.2f} % av värdena klipps")
    return np.clip(X, 0, 1).astype(np.float32), np.concatenate(y).astype(np.int64)


arrays = {}
for split, records in [("train", lib.train_records), ("val", lib.val_records)] + ([("test", lib.test_records)] if RUN_TEST else []):
    arrays[f"X_{split}"], arrays[f"y_{split}"] = build(records)

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
np.savez(OUT_FILE, **arrays)
print(f"Sparade {os.path.abspath(OUT_FILE)}")
