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

window_size = 100
offset = 0

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(HERE, "results", "cnn_metrics.csv")
MODELS_DIR = os.path.join(HERE, "models")

train_dataset = lib.MITBIHDataset(record_list=lib.train_records, window_size=window_size, offset=offset)
val_dataset = lib.MITBIHDataset(record_list=lib.val_records, window_size=window_size, offset=offset)

X_train = torch.tensor(train_dataset.beats).unsqueeze(1)
y_train = torch.tensor(train_dataset.labels)
X_val = torch.tensor(val_dataset.beats).unsqueeze(1)
y_val = val_dataset.labels

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Conv1d(1, 8, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool1d(2),
    nn.Conv1d(8, 16, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool1d(2),
    nn.Flatten(),
    nn.Linear(16 * 25, 4),
)

# Samma klassviktning som i MultilayerPerceptron.py
class_counts = np.bincount(train_dataset.labels)
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
        print(f"Epok {epoch:3d} | Val F1 macro: {val_f1 * 100:.2f}%", flush=True)

print(f"\nBästa epok: {best_epoch}, Val F1 macro: {best_f1 * 100:.2f}%")

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    y_pred = model(X_val).argmax(dim=1).numpy()

print(classification_report(y_val, y_pred, target_names=lib.CLASS_NAMES, digits=4, zero_division=0))
print(confusion_matrix(y_val, y_pred))

result = {
    "model": "Small CNN",
    "features": "raw",
    "split": "val",
    "accuracy": accuracy_score(y_val, y_pred),
    "f1_macro": f1_score(y_val, y_pred, average='macro'),
}
precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, labels=[0, 1, 2, 3], zero_division=0)
for i, cls in enumerate(["N", "S", "V", "F"]):
    result[f"precision_{cls}"] = precision[i]
    result[f"recall_{cls}"] = recall[i]
    result[f"f1_{cls}"] = f1[i]

pd.DataFrame([result]).round(4).to_csv(RESULTS_FILE, index=False)
torch.save(best_state, os.path.join(MODELS_DIR, f"SmallCNN-valF1-{best_f1:.4f}-seed{SEED}.pth"))
