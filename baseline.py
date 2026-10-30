# ------------------ TASK 4.1 Baseline-----------------
# Logistic Regression, Random Forest, Support Vector Machine, CNN and tiny nn

import os
import numpy as np
import pandas as pd
import lib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

SEED = 42
np.random.seed(SEED)
window_size = 100
offset = 0

# Sätt till True först när valet av baslinje är låst
RUN_TEST = True

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(HERE, f"features_cache_w{window_size}_o{offset}.npz")
RESULTS_FILE = os.path.join(HERE, "results", "baseline_metrics.csv")


def build_cache():
    print("Bygger features (tar ca 1,5 min)...")
    train = lib.MITBIHFeatureDataset(record_list=lib.train_records, window_size=window_size, offset=offset)
    val = lib.MITBIHFeatureDataset(record_list=lib.val_records, window_size=window_size, offset=offset,
                                   templates=train.templates, feature_mean=train.feature_mean, feature_std=train.feature_std)
    test = lib.MITBIHFeatureDataset(record_list=lib.test_records, window_size=window_size, offset=offset,
                                    templates=train.templates, feature_mean=train.feature_mean, feature_std=train.feature_std)
    np.savez(CACHE_FILE,
             X_train=train.features, y_train=train.labels,
             X_val=val.features, y_val=val.labels,
             X_test=test.features, y_test=test.labels,
             feature_names=train.feature_names)


if not os.path.exists(CACHE_FILE):
    build_cache()

data = np.load(CACHE_FILE)
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]
feature_names = list(data["feature_names"])

print(X_train.shape, X_val.shape, X_test.shape)   # t.ex. (41246, 44) (9763, 44) (49700, 44)

chosen = ['template_n_distance', 'template_v_distance', 'pre_rr', 'post_rr', 'compensatory_pause_ratio']
idx = [feature_names.index(name) for name in chosen]
X_train_5, X_val_5, X_test_5 = X_train[:, idx], X_val[:, idx], X_test[:, idx]


def evaluate(model, X, y, name):
    y_pred = model.predict(X)
    print(f"\n--- {name} ---")
    print(classification_report(y, y_pred, target_names=lib.CLASS_NAMES, digits=4, zero_division=0))
    print(confusion_matrix(y, y_pred))

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


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED, n_jobs=-1),
    "SVM": LinearSVC(class_weight='balanced', random_state=SEED),
}

results = []
for n_features, X_tr, X_v, X_te in [(44, X_train, X_val, X_test), (5, X_train_5, X_val_5, X_test_5)]:
    for name, model in models.items():
        model.fit(X_tr, y_train)

        result = evaluate(model, X_v, y_val, f"{name}, {n_features} features (val)")
        results.append({"model": name, "features": n_features, "split": "val", **result})

        if RUN_TEST:
            result = evaluate(model, X_te, y_test, f"{name}, {n_features} features (test)")
            results.append({"model": name, "features": n_features, "split": "test", **result})

os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
pd.DataFrame(results).round(4).to_csv(RESULTS_FILE, index=False)
print(f"\nSparade resultat till {RESULTS_FILE}")
