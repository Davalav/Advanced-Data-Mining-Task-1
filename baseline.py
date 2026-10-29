# ------------------ TASK 4.1 Baseline-----------------
# Logistic Regression, Random Forest, Support Vector Machine, CNN and tiny nn

import os
import numpy as np
import lib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

SEED = 42
np.random.seed(SEED)
window_size = 100
offset = 0
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"features_cache_w{window_size}_o{offset}.npz")


def load_features():
    """Läser features från cache om den finns, annars byggs de och sparas."""
    if os.path.exists(CACHE_FILE):
        print(f"Laddar features från {CACHE_FILE}")
        data = np.load(CACHE_FILE)
        return (data["X_train"], data["y_train"],
                data["X_val"], data["y_val"],
                data["X_test"], data["y_test"],
                list(data["feature_names"]))

    print("Ingen cache hittad, bygger features (tar ca 1,5 min)...")
    train_dataset = lib.MITBIHFeatureDataset(record_list=lib.train_records, window_size=window_size, offset=offset)
    val_dataset   = lib.MITBIHFeatureDataset(record_list=lib.val_records, window_size=window_size, offset=offset,
                                             templates=train_dataset.templates,
                                             feature_mean=train_dataset.feature_mean,
                                             feature_std=train_dataset.feature_std)
    test_dataset  = lib.MITBIHFeatureDataset(record_list=lib.test_records, window_size=window_size, offset=offset,
                                             templates=train_dataset.templates,
                                             feature_mean=train_dataset.feature_mean,
                                             feature_std=train_dataset.feature_std)

    np.savez(CACHE_FILE,
             X_train=train_dataset.features, y_train=train_dataset.labels,
             X_val=val_dataset.features,     y_val=val_dataset.labels,
             X_test=test_dataset.features,   y_test=test_dataset.labels,
             feature_names=np.array(train_dataset.feature_names))
    print(f"Sparade features till {CACHE_FILE}")

    return (train_dataset.features, train_dataset.labels,
            val_dataset.features, val_dataset.labels,
            test_dataset.features, test_dataset.labels,
            train_dataset.feature_names)


X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_features()

print(X_train.shape, X_val.shape, X_test.shape)   # t.ex. (41246, 44) (9763, 44) (49700, 44)

chosen = ['template_n_distance', 'template_v_distance', 'pre_rr', 'post_rr', 'compensatory_pause_ratio']
idx = [feature_names.index(name) for name in chosen]
X_train_5, X_val_5, X_test_5 = X_train[:, idx], X_val[:, idx], X_test[:, idx]


def evaluate(model, X, y, name):
    """Skriver ut accuracy, F1 macro och per-klass-resultat för en tränad modell."""
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc * 100:.2f}% - F1 Macro: {f1 * 100:.2f}%")
    print(classification_report(y, y_pred, target_names=lib.CLASS_NAMES, digits=4, zero_division=0))
    print("Confusion matrix (rader = sann klass, kolumner = predikterad):")
    print(confusion_matrix(y, y_pred))
    return acc, f1


# ------------------ Logistic Regression -----------------
# Utvärderas på validering (DS1). Testdatan (DS2) körs först när baslinjen är låst.
for X_tr, X_v, label in [(X_train, X_val, "44 features"), (X_train_5, X_val_5, "5 features")]:
    log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)
    log_reg.fit(X_tr, y_train)
    evaluate(log_reg, X_v, y_val, f"Logistic Regression, {label} (validering)")
