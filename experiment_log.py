import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_FILE = os.path.join(HERE, "results", "experiments.csv")
COMPARISON_FILE = os.path.join(HERE, "results", "seed_comparison.csv")

SETTINGS = ["model", "features", "lr", "batch_size", "epochs", "class_weights", "sampler"]


def save_experiment(result):
    experiments = pd.DataFrame([result])

    # Samma modell, inställningar och seed ersätter den gamla raden
    if os.path.exists(EXPERIMENTS_FILE):
        old = pd.read_csv(EXPERIMENTS_FILE, dtype={"features": str})
        experiments = pd.concat([old, experiments], ignore_index=True)
        experiments = experiments.drop_duplicates(subset=SETTINGS + ["seed"], keep="last")

    experiments = experiments.sort_values(SETTINGS + ["seed"]).round(4)
    experiments.to_csv(EXPERIMENTS_FILE, index=False)

    stats = experiments.groupby(SETTINGS).agg(
        runs=("seed", "count"),
        f1_mean=("f1_macro", "mean"),
        f1_std=("f1_macro", "std"),
        accuracy_mean=("accuracy", "mean"),
        f1_N_mean=("f1_N", "mean"),
        f1_S_mean=("f1_S", "mean"),
        f1_V_mean=("f1_V", "mean"),
        f1_F_mean=("f1_F", "mean"),
        recall_S_mean=("recall_S", "mean"),
        recall_V_mean=("recall_V", "mean"),
        recall_F_mean=("recall_F", "mean"),
    )
    f1_per_seed = experiments.pivot_table(index=SETTINGS, columns="seed", values="f1_macro").add_prefix("f1_seed_")

    comparison = stats.join(f1_per_seed).round(4).sort_values("f1_mean", ascending=False).reset_index()
    comparison.to_csv(COMPARISON_FILE, index=False)
    print(f"\nSparade resultat till {EXPERIMENTS_FILE} och {COMPARISON_FILE}")
