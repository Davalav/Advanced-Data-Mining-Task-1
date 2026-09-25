import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import wfdb
import torchshow as ts
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import copy
from collections import Counter
import lib
from torch.utils.data import WeightedRandomSampler
import sys
start_time=time.time()

print(f"Python version: {torch.sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available? {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Initialize
USE_FEATURE_SELECTION = True
N_FEATURES_TO_SELECT = 5  # tune as needed; None = half of all features
input_dim = N_FEATURES_TO_SELECT
num_classes = 4
window_size=100
offset = 10

model = lib.BeatClassifierMLP(input_dim=input_dim, num_classes=num_classes).to(device)
if len(sys.argv) >1:
    model.load_state_dict(torch.load(sys.argv[1]))
else:
    raise Exception("Missing file argument")
start_time=time.time()

train_dataset = lib.MITBIHFeatureDataset(record_list=lib.train_records, window_size=window_size)
test_dataset   = lib.MITBIHFeatureDataset(record_list=lib.test_records, window_size=window_size, feature_mean=train_dataset.feature_mean, feature_std=train_dataset.feature_std,)
elapsed_time=time.time()-start_time
time_A=time.time()
print(f"Dataset init took {lib.format_seconds(elapsed_time)}")

# --- Feature selection (fit on train only, applied to both splits) ----

if USE_FEATURE_SELECTION:
    selected_indices, _ = lib.select_features(
        train_dataset,
        n_features_to_select=N_FEATURES_TO_SELECT,
        direction='forward',
        cv=3,
        scoring='f1_macro',
    )
    lib.apply_feature_selection(test_dataset, selected_indices)

    # If you build a separate test_dataset elsewhere, apply the same
    # selected_indices to it before evaluating - never re-fit selection
    # on val/test.

elapsed_time=time.time()-time_A
time_A=time.time()
print(f"Feature selection took {lib.format_seconds(elapsed_time)}")

test_loader   = DataLoader(test_dataset, batch_size=64, shuffle=False)

class_counts = np.bincount(train_dataset.labels, minlength=num_classes)
class_weights = torch.tensor(
    1.0 / np.sqrt(np.maximum(class_counts, 1)), dtype=torch.float32
)
class_weights = class_weights / class_weights.sum() * num_classes
class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

test_loss, test_acc   = lib.evaluate(model, test_loader, criterion, device)


test_f1 = lib.f1_calc(model,test_loader,device)

print(f"Final test | "
            f"Test Loss: {test_loss:.4f} - Acc: {test_acc * 100:.2f}% - F1 Macro {test_f1*100:.2f}%")

cm =lib.evaluate_and_plot_cm(model,test_loader,device, "Testing")

print(f"Total time: {lib.format_seconds(time.time()-start_time)}")