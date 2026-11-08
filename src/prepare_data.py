# ------------------ Prepare data -----------------
# Bygger feature-cachen: 44 handgjorda features per slag för train, val och test.
# Baslinjen och MLP:n använder alla features, train.py tar RR-värdena härifrån.
# Mallarna och normaliseringen beräknas bara på träningsdatan.

import os

import numpy as np

import dataset
from config import CONFIG

window_size = CONFIG["window_size"]
offset = CONFIG["offset"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_FILE = os.path.join(ROOT, f"features_cache_w{window_size}_o{offset}.npz")


def build_cache():
    print("Bygger features (tar ca 1,5 min)...")
    train = dataset.MITBIHFeatureDataset(record_list=dataset.train_records, window_size=window_size, offset=offset)
    val = dataset.MITBIHFeatureDataset(record_list=dataset.val_records, window_size=window_size, offset=offset,
                                       templates=train.templates, feature_mean=train.feature_mean, feature_std=train.feature_std)
    test = dataset.MITBIHFeatureDataset(record_list=dataset.test_records, window_size=window_size, offset=offset,
                                        templates=train.templates, feature_mean=train.feature_mean, feature_std=train.feature_std)
    np.savez(CACHE_FILE,
             X_train=train.features, y_train=train.labels,
             X_val=val.features, y_val=val.labels,
             X_test=test.features, y_test=test.labels,
             feature_names=train.feature_names)
    print(f"Sparade {CACHE_FILE}")


if __name__ == "__main__":
    build_cache()
