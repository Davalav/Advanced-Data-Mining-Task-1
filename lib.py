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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import copy
from collections import Counter


# Mapping MIT-BIH symbols to standard 5 AAMI classes
AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal / Non-ectopic
    'A': 1, 'a': 1, 'J': 1, 'S': 1,          # Supraventricular Ectopic
    'V': 2, 'E': 2,                          # Ventricular Ectopic
    'F': 3#,                                  # Fusion Beat
    #'/': 4, 'f': 4, 'Q': 4                   # Unknown / Paced / Artifacts
}
CLASS_NAMES = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)']#, 'Unknown (Q)']

directory = "../mit-bih-arrhythmia-database-1.0.0/"

def compute_class_weights(labels, num_classes=5):
    """
    Computes balanced class weights inversely proportional to class frequencies.
    
    Formula: weight_i = total_samples / (num_classes * count_i)
    """
    # Count occurrences of each class index
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    
    # Calculate inverse frequency weights
    # Prevent division by zero if a class has 0 samples
    weights = []
    for count in class_counts:
        if count > 0:
            w = total_samples / (num_classes * count)
        else:
            w = 1.0
        weights.append(w)
        
    weights = np.array(weights, dtype=np.float32)
    
    # Optional: Normalize weights so their mean equals 1.0
    weights = weights / np.mean(weights)
    
    return torch.tensor(weights, dtype=torch.float32)

def format_seconds(seconds: float) -> str:
    """
    Convert a float number of seconds into a human-readable string.
    Automatically scales from milliseconds up to days.
    """
    abs_seconds = abs(seconds)
    sign = "-" if seconds < 0 else ""

    if abs_seconds < 1e-3:
        return f"{sign}{abs_seconds * 1e6:.3f} µs"
    elif abs_seconds < 1:
        return f"{sign}{abs_seconds * 1e3:.3f} ms"
    elif abs_seconds < 60:
        return f"{sign}{abs_seconds:.3f} s"
    elif abs_seconds < 3600:
        return f"{sign}{abs_seconds / 60:.3f} min"
    elif abs_seconds < 86400:
        return f"{sign}{abs_seconds / 3600:.3f} h"
    else:
        return f"{sign}{abs_seconds / 86400:.3f} days"


class MITBIHDataset(Dataset):
    """
    PyTorch Dataset for MIT-BIH Arrhythmia Database.
    Extracts 1D ECG beat segments centered on annotated R-peaks.
    """
    def __init__(self, record_list, window_size=256, channel=0):
        """
        Args:
            record_list (list): List of record IDs, e.g., ['100', '101', '102'].
            window_size (int): Total signal length per window.
            channel (int): ECG lead channel (0 is typically Lead II).
        """
        self.window_size = window_size
        self.channel = channel
        self.beats = []
        self.labels = []
        
        self._load_data(record_list)

    def _load_data(self, record_list):
        half_window = self.window_size // 2
        
        for record_name in record_list:
            # Load raw signal and annotations
            record = wfdb.rdrecord(directory+record_name)
            annotation = wfdb.rdann(directory+record_name, 'atr')
            
            signal = record.p_signal[:, self.channel]
            
            # Extract valid beats
            for sample_idx, symbol in zip(annotation.sample, annotation.symbol):
                if symbol not in AAMI_MAPPING:
                    #print("WARNING: INVALID BEAT: ["+ str(sample_idx)+"] "+ symbol)
                    continue
                
                start_idx = sample_idx - half_window
                end_idx = sample_idx + half_window
                
                # Boundary check
                if start_idx >= 0 and end_idx < len(signal):
                    segment = signal[start_idx:end_idx]

                    # Apply Z-score normalization to each window individually
                    std = np.std(segment)
                    if std > 0:
                        segment = (segment - np.mean(segment)) / std
                    else:
                        segment = segment - np.mean(segment)
                    
                    self.beats.append(segment)
                    self.labels.append(AAMI_MAPPING[symbol])
                    
        # Convert lists to NumPy arrays
        self.beats = np.array(self.beats, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.beats)

    def __getitem__(self, idx):
        # Shape output for PyTorch 1D Convolution: [Channels, Sequence_Length]
        x = torch.tensor(self.beats[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
    
class ECG1DCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ECG1DCNN, self).__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            # Block 1: [Batch, 1, 256] -> [Batch, 32, 128]
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Block 2: [Batch, 32, 128] -> [Batch, 64, 64]
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Block 3: [Batch, 64, 64] -> [Batch, 128, 32]
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    test_loss = running_loss / total
    test_acc = correct / total
    return test_loss, test_acc

def count_parameters(model):
    print("Modules", "Parameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        print(f"{name}: {params}")
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params
def f1_calc(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Get class predictions (highest probability/logit index)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    return f1_score(all_targets, all_preds,average='macro')
    
    
def evaluate_and_plot_cm(model, dataloader, device):
    """
    Evaluates the model on the test set, prints a classification report,
    and displays an sklearn confusion matrix heatmap.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Get class predictions (highest probability/logit index)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 1. Print Detailed Metrics (Precision, Recall, F1-score)
    print("\n--- Classification Report ---")
    print(classification_report(
        all_targets, 
        all_preds, 
        target_names=CLASS_NAMES, 
        digits=4, 
        zero_division=0
    ))

    # 2. Compute Confusion Matrix via scikit-learn
    cm = confusion_matrix(all_targets, all_preds, labels=range(len(CLASS_NAMES)), normalize='true')

    # 3. Plot Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='.2%', 
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title('MIT-BIH Classification Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return cm

def log_plot_balance(dataset):
    counts = Counter(dataset.labels)
    #print(counts[0])
    counts = [counts[i] for i in range(len(CLASS_NAMES))]
    #print(counts[0])
    count_sum = sum(counts)
    for i in range(len(counts)):
        counts[i]=counts[i]*100/count_sum
    print(counts)
    # Skapa histogrammet
    plt.bar(CLASS_NAMES,counts, color='skyblue', edgecolor='black', log=True)
    # Lägg till titlar och etiketter
    plt.title('Frequency of labels')
    plt.xlabel('Labels')
    plt.ylabel('Procent')
    plt.show()

# 1. Define train and validation record splits
train_val_records = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230']
test_records   = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']

# best 80% split
#train_records = ['101', '106', '108', '109', '114', '115', '116', '118', '119', '122', '124', '203', '207', '208', '209', '215', '220', '230']
#val_records =['112', '201', '205', '223']


# NEW best 80% split
train_records = ['101', '106', '108', '109', '114', '115', '116', '118', '119', '122', '124', '203', '207', '208', '209', '215', '220', '230']
val_records = ['112', '201', '205', '223']

# best 90% split
#train_records = ['101', '106', '108', '109', '112', '114', '115', '118', '119', '122', '124', '201', '203', '205', '208', '209', '215', '220', '223', '230']
#val_records = ['116', '207']

# NEW best 90% split
#train_records = ['101', '106', '108', '109', '112', '114', '115', '118', '119', '122', '124', '201', '203', '205', '208', '209', '215', '220', '223', '230']
#val_records = ['116', '207']