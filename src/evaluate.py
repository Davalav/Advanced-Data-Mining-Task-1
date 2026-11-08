# ------------------ Final evaluation -----------------
# Baslinjen (Random Forest, 44 features) och den valda CNN:en i FP32 och INT8 på samma slag.
# Sparar mått, confusion matrix och resultat per inspelning (patient).

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

import dataset
import model as cnn
from config import CONFIG
from prepare_data import CACHE_FILE

# Sätt till True först när allt annat är låst, DS2 ska bara köras en gång
RUN_TEST = True

SEEDS = CONFIG["seeds"]
FIGURE_SEED = CONFIG["deployed_seed"]                # modellen som visas i confusion matrix och per inspelning, samma som i deployment.py
FILTERS = CONFIG["filters"]
window_size = CONFIG["window_size"]
offset = CONFIG["offset"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
split = "test" if RUN_TEST else "val"
records = dataset.test_records if RUN_TEST else dataset.val_records
suffix = "" if RUN_TEST else "_val"   # slutresultaten på DS2 får de namn som slutpaketet kräver
METRICS_FILE = os.path.join(ROOT, "results", f"metrics{suffix}.csv")
CONFUSION_FILE = os.path.join(ROOT, "results", f"confusion_matrix{suffix}.csv")
CONFUSION_PLOT = os.path.join(ROOT, "results", f"confusion_matrix{suffix}.png")
RECORD_FILE = os.path.join(ROOT, "results", f"per_record{suffix}.csv")
CLASSES = ["N", "S", "V", "F"]

# ---- Indata till CNN:en: fönster och RR från annoteringarna, som i quantize.py ----
rr_mean, rr_std = dataset.train_rr_stats(window_size, offset)
X, y, beat_records = [], [], []
for record_name in records:
    record = wfdb.rdrecord(dataset.directory + record_name)
    annotation = wfdb.rdann(dataset.directory + record_name, 'atr')
    signal = record.p_signal[:, 0]
    beat_samples, labels = dataset.kept_beats(annotation, len(signal), window_size, offset)
    rr = (dataset.rr_intervals(beat_samples, dataset.annotated_peaks(annotation), record.fs) - rr_mean) / rr_std
    windows = dataset.beat_windows(dataset.filter_signal(signal, record.fs), beat_samples, window_size, offset)
    X.append(np.concatenate([windows[:, None, :], np.repeat(rr[:, :, None], window_size, axis=2)], axis=1))
    y.append(labels)
    beat_records += [record_name] * len(labels)
X = torch.tensor(np.concatenate(X), dtype=torch.float32)
y = np.concatenate(y)
beat_records = np.array(beat_records)

# ---- Baslinjen på de cachade features, samma inställningar som i baseline.py ----
cache = np.load(CACHE_FILE)
assert (cache[f"y_{split}"] == y).all(), "Feature-cachen och CNN-indatan har inte samma slag"
baseline = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
baseline.fit(cache["X_train"], cache["y_train"])

filters_name = "-".join(map(str, FILTERS))
predictions = [("Random Forest (44 features)", "-", baseline.predict(cache[f"X_{split}"]))]
for seed in SEEDS:
    fp32_file, = glob.glob(os.path.join(MODELS_DIR, f"SmallCNN-valF1-*-seed{seed}-lr0.02-cw0-sampler1-rr1-filters{filters_name}.pth"))
    int8_file, = glob.glob(os.path.join(MODELS_DIR, f"SmallCNN-INT8-valF1-*-seed{seed}-filters{filters_name}.pth"))
    for name, model in [("CNN FP32", cnn.load_cnn_fp32(fp32_file, FILTERS, window_size)),
                        ("CNN INT8", cnn.load_cnn_int8(int8_file, FILTERS, window_size))]:
        with torch.no_grad():
            y_pred = torch.cat([model(batch).argmax(dim=1) for batch in X.split(256)]).numpy()
        predictions.append((name, seed, y_pred))


def metrics(y_true, y_pred):
    result = {
        "beats": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average='macro', zero_division=0),
    }
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0)
    for i, cls in enumerate(CLASSES):
        result[f"precision_{cls}"] = precision[i]
        result[f"recall_{cls}"] = recall[i]
        result[f"f1_{cls}"] = f1[i]
    return result


# ---- Mått per modell och seed, med medelvärde över seeds sist ----
rows = [{"model": name, "seed": seed, "split": split, **metrics(y, y_pred)} for name, seed, y_pred in predictions]
per_seed = pd.DataFrame(rows)
grouped = per_seed[per_seed["seed"] != "-"].drop(columns=["seed", "split"]).groupby("model", sort=False)
means = grouped.mean().reset_index()
means["f1_std"] = grouped["f1_macro"].std().values
means["seed"] = "mean"
means["split"] = split
pd.concat([per_seed, means], ignore_index=True).round(4).to_csv(METRICS_FILE, index=False)

# ---- Confusion matrix och utskrift för baslinjen och seed FIGURE_SEED ----
shown = [(name, y_pred) for name, seed, y_pred in predictions if seed in ("-", FIGURE_SEED)]
matrices = []
for name, y_pred in shown:
    print(f"\n--- {name} ({split}) ---")
    print(classification_report(y, y_pred, labels=[0, 1, 2, 3], target_names=dataset.CLASS_NAMES, digits=4, zero_division=0))
    matrix = confusion_matrix(y, y_pred, labels=[0, 1, 2, 3])
    print(matrix)
    matrices.append(matrix)
confusion_rows = [{"model": name, "true": CLASSES[i], **{f"predicted_{c}": matrix[i, j] for j, c in enumerate(CLASSES)}}
                  for (name, _), matrix in zip(shown, matrices) for i in range(len(CLASSES))]
pd.DataFrame(confusion_rows).to_csv(CONFUSION_FILE, index=False)

fig, axes = plt.subplots(1, len(shown), figsize=(4.2 * len(shown), 4.2), facecolor="#fcfcfb")
for ax, (name, _), matrix in zip(axes, shown, matrices):
    share = matrix / matrix.sum(axis=1, keepdims=True)   # färgen visar andel av varje rätt klass
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{matrix[i, j]:,}\n{share[i, j] * 100:.1f} %", ha="center", va="center", fontsize=8,
                    color="#ffffff" if share[i, j] > 0.6 else "#0b0b0b")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted", color="#52514e")
    ax.set_title(name if "Forest" in name else f"{name}, seed {FIGURE_SEED}", loc="left", fontsize=10, color="#0b0b0b")
axes[0].set_ylabel("True", color="#52514e")
fig.suptitle(f"Confusion matrices on {split} data (count and share of each true class)", x=0.01, ha="left", color="#0b0b0b")
fig.tight_layout()
fig.savefig(CONFUSION_PLOT, dpi=150, facecolor="#fcfcfb")

# ---- Resultat per inspelning, för skillnader mellan patienter ----
record_rows = []
for name, y_pred in shown:
    for record_name in records:
        mask = beat_records == record_name
        counts = np.bincount(y[mask], minlength=len(CLASSES))
        row = {"model": name, "record": record_name, **{f"beats_{c}": counts[i] for i, c in enumerate(CLASSES)},
               "accuracy": accuracy_score(y[mask], y_pred[mask])}
        recall = precision_recall_fscore_support(y[mask], y_pred[mask], labels=[0, 1, 2, 3], zero_division=0)[1]
        for i, cls in enumerate(CLASSES):
            row[f"recall_{cls}"] = recall[i] if counts[i] > 0 else np.nan   # tomt när klassen saknas i inspelningen
        record_rows.append(row)
pd.DataFrame(record_rows).round(4).to_csv(RECORD_FILE, index=False)

print("\n" + means[["model", "f1_macro", "f1_std", "accuracy"]].round(4).to_string(index=False))
print(per_seed[per_seed["seed"] == "-"][["model", "f1_macro", "accuracy"]].round(4).to_string(index=False))
print(f"\nSparade {METRICS_FILE}, {CONFUSION_FILE}, {CONFUSION_PLOT} och {RECORD_FILE}")
