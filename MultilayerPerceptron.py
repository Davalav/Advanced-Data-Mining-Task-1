import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

import lib

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(HERE, "features_cache_w100_o0.npz"))

chosen = ['template_n_distance', 'template_v_distance', 'pre_rr', 'post_rr', 'compensatory_pause_ratio']
feature_names = list(data["feature_names"])
idx = [feature_names.index(name) for name in chosen]

X_train = torch.tensor(data["X_train"][:, idx], dtype=torch.float32)
y_train = torch.tensor(data["y_train"], dtype=torch.long)
X_val = torch.tensor(data["X_val"][:, idx], dtype=torch.float32)
y_val = data["y_val"]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Linear(5, 32),
    nn.ReLU(),
    nn.Linear(32, 4),
)

# Samma klassviktning som i train-feature.py
class_counts = np.bincount(data["y_train"])
class_weights = torch.tensor(1.0 / np.sqrt(class_counts), dtype=torch.float32)
class_weights = class_weights / class_weights.sum() * len(class_counts)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
epochs = 100

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
        print(f"Epok {epoch:3d} | Val F1 macro: {val_f1 * 100:.2f}%")

print(f"\nBästa epok: {best_epoch}, Val F1 macro: {best_f1 * 100:.2f}%")

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    y_pred = model(X_val).argmax(dim=1).numpy()

print(classification_report(y_val, y_pred, target_names=lib.CLASS_NAMES, digits=4, zero_division=0))
print(confusion_matrix(y_val, y_pred))

result = {
    "model": "MLP",
    "features": len(idx),
    "split": "val",
    "accuracy": accuracy_score(y_val, y_pred),
    "f1_macro": f1_score(y_val, y_pred, average='macro'),
}
precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, labels=[0, 1, 2, 3], zero_division=0)
for i, cls in enumerate(["N", "S", "V", "F"]):
    result[f"precision_{cls}"] = precision[i]
    result[f"recall_{cls}"] = recall[i]
    result[f"f1_{cls}"] = f1[i]

pd.DataFrame([result]).round(4).to_csv(os.path.join(HERE, "results", "mlp_metrics.csv"), index=False)
torch.save(best_state, os.path.join(HERE, "models", f"MLP-valF1-{best_f1:.4f}-seed{SEED}.pth"))
