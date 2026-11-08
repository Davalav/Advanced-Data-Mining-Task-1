# ------------------ Training of the CNN -----------------
# python src/train.py SEED LR CLASS_WEIGHTS SAMPLER RR FILTERS, e.g. python src/train.py 42 0.02 0 1 1 16,32,64
# Arguments that are left out are taken from configs/final_model.json
import copy
import sys
import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

import dataset
import model as cnn
from config import CONFIG
from prepare_data import CACHE_FILE
from experiment_log import save_experiment

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
LR = float(sys.argv[2]) if len(sys.argv) > 2 else CONFIG["lr"]
USE_CLASS_WEIGHTS = sys.argv[3] == "1" if len(sys.argv) > 3 else CONFIG["class_weights"]
USE_SAMPLER = sys.argv[4] == "1" if len(sys.argv) > 4 else CONFIG["sampler"]
USE_RR = sys.argv[5] == "1" if len(sys.argv) > 5 else CONFIG["rr_input"]
FILTERS = [int(f) for f in sys.argv[6].split(",")] if len(sys.argv) > 6 else CONFIG["filters"]
np.random.seed(SEED)
torch.manual_seed(SEED)

window_size = CONFIG["window_size"]
offset = CONFIG["offset"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")

train_dataset = dataset.MITBIHDataset(record_list=dataset.train_records, window_size=window_size, offset=offset)
val_dataset = dataset.MITBIHDataset(record_list=dataset.val_records, window_size=window_size, offset=offset)

X_train = torch.tensor(train_dataset.beats).unsqueeze(1)
y_train = torch.tensor(train_dataset.labels)
X_val = torch.tensor(val_dataset.beats).unsqueeze(1)
y_val = val_dataset.labels

if USE_RR:
    cache = np.load(CACHE_FILE)
    feature_names = list(cache["feature_names"])
    rr_idx = [feature_names.index("pre_rr"), feature_names.index("post_rr")]

    def add_rr_channels(X, features):
        rr = torch.tensor(features[:, rr_idx], dtype=torch.float32)
        rr = rr.unsqueeze(2).repeat(1, 1, X.shape[2])
        return torch.cat([X, rr], dim=1)

    X_train = add_rr_channels(X_train, cache["X_train"])
    X_val = add_rr_channels(X_val, cache["X_val"])

class_counts = np.bincount(train_dataset.labels)

if USE_SAMPLER:
    sample_weights = 1.0 / np.sqrt(class_counts[train_dataset.labels])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True,
                                    generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=CONFIG["batch_size"], sampler=sampler)
else:
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=CONFIG["batch_size"], shuffle=True)

model = cnn.build_cnn(FILTERS, in_channels=X_train.shape[1], window_size=window_size)

# Samma klassviktning som i train_mlp.py
class_weights = torch.tensor(1.0 / np.sqrt(class_counts), dtype=torch.float32)
class_weights = class_weights / class_weights.sum() * len(class_counts)

criterion = nn.CrossEntropyLoss(weight=class_weights if USE_CLASS_WEIGHTS else None)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=CONFIG["momentum"])
epochs = CONFIG["epochs"]

start_time = time.time()
best_f1 = -1
best_epoch = -1
best_state = None

for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_val).argmax(dim=1).numpy()
    val_f1 = f1_score(y_val, y_pred, average='macro')

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())

    if epoch % 10 == 0:
        print(f"Epok {epoch:3d} | Val F1 macro: {val_f1 * 100:.2f}%", flush=True)

print(f"\nBästa epok: {best_epoch}, Val F1 macro: {best_f1 * 100:.2f}%")

train_seconds = time.time() - start_time
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    y_pred = model(X_val).argmax(dim=1).numpy()

print(classification_report(y_val, y_pred, target_names=dataset.CLASS_NAMES, digits=4, zero_division=0))
print(confusion_matrix(y_val, y_pred))

result = {
    "model": "Small CNN",
    "features": "raw+rr" if USE_RR else "raw",
    "split": "val",
    "seed": SEED,
    "lr": optimizer.defaults["lr"],
    "batch_size": train_loader.batch_size,
    "epochs": epochs,
    "class_weights": USE_CLASS_WEIGHTS,
    "sampler": USE_SAMPLER,
    "n_parameters": sum(p.numel() for p in model.parameters()),
    "best_epoch": best_epoch,
    "train_seconds": train_seconds,
    "accuracy": accuracy_score(y_val, y_pred),
    "f1_macro": f1_score(y_val, y_pred, average='macro'),
}
precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, labels=[0, 1, 2, 3], zero_division=0)
for i, cls in enumerate(["N", "S", "V", "F"]):
    result[f"precision_{cls}"] = precision[i]
    result[f"recall_{cls}"] = recall[i]
    result[f"f1_{cls}"] = f1[i]

save_experiment(result)
torch.save(best_state, os.path.join(MODELS_DIR, f"SmallCNN-valF1-{best_f1:.4f}-seed{SEED}-lr{LR}-cw{int(USE_CLASS_WEIGHTS)}-sampler{int(USE_SAMPLER)}-rr{int(USE_RR)}-filters{'-'.join(map(str, FILTERS))}.pth"))
