# ------------------ Noise robustness -----------------
# Lägger NSTDB-brus på EKG-signalen och testar FP32- och INT8-modellen

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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
window_size = 100
offset = 0

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
NOISE_DIR = os.path.join(HERE, "..", "mit-bih-noise-stress-test-database-1.0.0") + os.sep

split = "test" if RUN_TEST else "val"
records = lib.test_records if RUN_TEST else lib.val_records
RESULTS_FILE = os.path.join(HERE, "results", f"noise_results_{split}.csv")
SUMMARY_FILE = os.path.join(HERE, "results", f"noise_summary_{split}.csv")
MARKDOWN_FILE = os.path.join(HERE, "results", f"noise_summary_{split}.md")
PLOT_FILE = os.path.join(HERE, "results", f"noise_results_{split}.png")

rr_mean, rr_std = lib.train_rr_stats(window_size, offset)

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
            signal = lib.add_noise(signal, noise_signals[noise_type], snr_db)
        beat_samples, labels = lib.kept_beats(annotation, len(signal), window_size, offset)

        if rr_source == "detector":
            peaks = Detectors(fs).pan_tompkins_detector(signal)
        else:
            peaks = lib.annotated_peaks(annotation)
        found.append(lib.found_by_detector(beat_samples, peaks, fs))

        rr = (lib.rr_intervals(beat_samples, peaks, fs) - rr_mean) / rr_std
        rr = np.nan_to_num(rr)   # inga toppar alls: medelvärdet används
        rr_channels = np.repeat(rr[:, :, None], window_size, axis=2)

        windows = lib.beat_windows(lib.filter_signal(signal, fs), beat_samples, window_size, offset)
        X.append(np.concatenate([windows[:, None, :], rr_channels], axis=1))
        y.append(labels)

    return torch.tensor(np.concatenate(X), dtype=torch.float32), np.concatenate(y), np.concatenate(found).mean()


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
    models += [("FP32", seed, lib.load_cnn_fp32(fp32_file, FILTERS, window_size)),
               ("INT8", seed, lib.load_cnn_int8(int8_file, FILTERS, window_size))]

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

per_seed = pd.DataFrame(results)

# Medelvärde över seeds läggs sist i samma CSV
group = ["noise", "snr_db", "rr_source", "format"]
grouped = per_seed.drop(columns=["seed", "split"]).groupby(group, sort=False, dropna=False)
means = grouped.mean().reset_index()
means["f1_std"] = grouped["f1_macro"].std().values
means["seed"] = "mean"
means["split"] = split

pd.concat([per_seed, means], ignore_index=True).round(4).to_csv(RESULTS_FILE, index=False)
print(f"\nSparade resultat till {RESULTS_FILE}")


# ---- Kort sammanfattning för README och rapport ----
def condition_name(noise_type, snr_db, rr_source):
    if noise_type == "clean":
        return "Clean, annotation RR" if rr_source == "annotations" else "Clean"
    return f"{noise_type} {int(snr_db)} dB"


summary = []
for (noise_type, snr_db, rr_source), rows in per_seed.groupby(["noise", "snr_db", "rr_source"], sort=False, dropna=False):
    row = {"condition": condition_name(noise_type, snr_db, rr_source), "beats_found": rows["beats_found"].iloc[0] * 100}
    for name, f1 in rows.groupby("format", sort=False)["f1_macro"]:
        row[f"{name}_f1_mean"] = f1.mean() * 100
        row[f"{name}_f1_std"] = f1.std() * 100
        row[f"{name}_f1_best"] = f1.max() * 100
    summary.append(row)
summary = pd.DataFrame(summary).round(2)
summary.to_csv(SUMMARY_FILE, index=False)

lines = [f"F1 macro (%) on {split} data, mean ± std and best of seeds {', '.join(map(str, SEEDS))}. "
         "RR is computed with a Pan-Tompkins detector except in the first row.",
         "",
         "| Condition | FP32 mean ± std | FP32 best | INT8 mean ± std | INT8 best | Beats found by detector |",
         "|---|---|---|---|---|---|"]
for row in summary.itertuples():
    lines.append(f"| {row.condition} | {row.FP32_f1_mean:.2f} ± {row.FP32_f1_std:.2f} | {row.FP32_f1_best:.2f} "
                 f"| {row.INT8_f1_mean:.2f} ± {row.INT8_f1_std:.2f} | {row.INT8_f1_best:.2f} | {row.beats_found:.2f} % |")
with open(MARKDOWN_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

# Graf: F1 mot brusnivå, en linje per brustyp
COLORS = {"bw": "#2a78d6", "em": "#eb6834", "ma": "#1baf7a"}
NOISE_NAMES = {"bw": "Baseline wander (bw)", "em": "Electrode motion (em)", "ma": "Muscle artifact (ma)"}
x_labels = ["Clean"] + [f"{snr_db} dB" for snr_db in SNR_DB]
x = np.arange(len(x_labels))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, facecolor="#fcfcfb")
for ax, name in zip(axes, ["FP32", "INT8"]):
    rows = per_seed[per_seed["format"] == name]
    clean = rows[(rows["noise"] == "clean") & (rows["rr_source"] == "detector")]["f1_macro"] * 100
    reference = rows[rows["rr_source"] == "annotations"]["f1_macro"].mean() * 100

    ax.axhline(reference, color="#898781", linestyle="--", linewidth=1)
    ax.text(x[-1] + 0.4, reference + 0.3, "Clean, annotation RR", ha="right", va="bottom", color="#52514e", fontsize=9)

    for noise_type in NOISE_TYPES:
        values = [clean] + [rows[(rows["noise"] == noise_type) & (rows["snr_db"] == snr_db)]["f1_macro"] * 100
                            for snr_db in SNR_DB]
        mean = np.array([v.mean() for v in values])
        std = np.array([v.std() for v in values])
        ax.fill_between(x, mean - std, mean + std, color=COLORS[noise_type], alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=COLORS[noise_type], linewidth=2, marker="o", markersize=6, label=NOISE_NAMES[noise_type])
        ax.text(x[-1] + 0.1, mean[-1], noise_type, va="center", color="#0b0b0b", fontsize=9)

    ax.set_title(name, loc="left", color="#0b0b0b")
    ax.set_xticks(x, x_labels)
    ax.set_xlim(-0.2, x[-1] + 0.4)
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#898781")

axes[0].set_ylabel("F1 macro (%)", color="#52514e")
axes[1].legend(frameon=False, loc="lower left", fontsize=9)
fig.suptitle(f"Noise robustness on {split} data (mean ± std over {len(SEEDS)} seeds)", x=0.01, ha="left", color="#0b0b0b")
fig.tight_layout()
fig.savefig(PLOT_FILE, dpi=150, facecolor="#fcfcfb")
print(f"Sparade sammanfattning till {SUMMARY_FILE}, {MARKDOWN_FILE} och {PLOT_FILE}")
