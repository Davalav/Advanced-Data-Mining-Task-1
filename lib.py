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
import scipy.signal as sp


# Specify cutoff in Hertz
lpf_cutoff = 0.5 
hpf_cutoff = 20

# 2. Funktion för Bandpassfilter (Högpass + Lågpass kombinerat)
def butter_bandpass_filter(data, cutoff_low, cutoff_high, fs, order=4):
    nyq = 0.5 * fs
    # Normalisera båda brytfrekvenserna som en lista [low, high]
    normal_cutoff = [cutoff_low / nyq, cutoff_high / nyq]

    # btype='bandpass' skapar både högpass och lågpass samtidigt
    sos = sp.butter(order, normal_cutoff, btype="bandpass", analog=False, output="sos")

    # Filtrera med nollfasförskjutning
    filtered_data = sp.sosfiltfilt(sos, data)
    return filtered_data


def notch_filter(data, notch_freq,fs, quality_factor=30.0):
    # Calculate filter coefficients
    b, a = sp.iirnotch(notch_freq, quality_factor, fs)

    # 3. Apply the filter using zero-phase filtering (filtfilt) to prevent time shifts
    filtered_signal = sp.filtfilt(b, a, data)
    return filtered_signal

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
    def __init__(self, record_list, window_size=256, channel=0, offset=0):
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
        self.offset = offset
        
        self._load_data(record_list)

    def _load_data(self, record_list):
        half_window = self.window_size // 2
        
        for record_name in record_list:
            # Load raw signal and annotations
            record = wfdb.rdrecord(directory+record_name)
            annotation = wfdb.rdann(directory+record_name, 'atr')
            
            signal = record.p_signal[:, self.channel]
            signal = butter_bandpass_filter(signal,lpf_cutoff,hpf_cutoff,record.fs)
            signal = notch_filter(signal, 50, record.fs)

            last_sample_idx=-self.window_size
            # Extract valid beats
            for sample_idx, symbol in zip(annotation.sample, annotation.symbol):
                if symbol not in AAMI_MAPPING:
                    #print("WARNING: INVALID BEAT: ["+ str(sample_idx)+"] "+ symbol)
                    continue
                if last_sample_idx + self.window_size > sample_idx:
                    print("Too close beats")
                start_idx = sample_idx - half_window + self.offset
                end_idx = sample_idx + half_window + self.offset
                last_sample_idx = sample_idx
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
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ECG_model_nature(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes=6,
        filter_length=64,
        kernel_size=16,
        drop_rate=0.2
    ):
        super(ECG1DCNN, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.filter_length = filter_length
        self.kernel_size = kernel_size
        self.drop_rate = drop_rate

        # ============================================================
        # Block 1: First convolutional residual block
        # ============================================================
        self.first_block = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=filter_length,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            nn.BatchNorm1d(filter_length),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=filter_length,
                out_channels=filter_length,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            nn.BatchNorm1d(filter_length),
            nn.ReLU(),

            nn.Dropout(drop_rate),

            nn.Conv1d(
                in_channels=filter_length,
                out_channels=filter_length,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            )
        )

        # ============================================================
        # Block 2: Main residual blocks
        # ============================================================
        self.main_blocks = nn.Sequential(
            # Block 0: 64 -> 64, stride 2
            ResidualBlock(
                in_channels=filter_length,
                out_channels=filter_length,
                kernel_size=kernel_size,
                stride=2,
                drop_rate=drop_rate
            ),

            # Block 1: 64 -> 64, stride 1
            ResidualBlock(
                in_channels=filter_length,
                out_channels=filter_length,
                kernel_size=kernel_size,
                stride=1,
                drop_rate=drop_rate
            ),

            # Block 2: 64 -> 64, stride 2
            ResidualBlock(
                in_channels=filter_length,
                out_channels=filter_length,
                kernel_size=kernel_size,
                stride=2,
                drop_rate=drop_rate
            ),

            # Block 3: 64 -> 64, stride 1
            ResidualBlock(
                in_channels=filter_length,
                out_channels=filter_length,
                kernel_size=kernel_size,
                stride=1,
                drop_rate=drop_rate
            ),

            # Block 4: 64 -> 128, stride 2
            ResidualBlock(
                in_channels=filter_length,
                out_channels=filter_length * 2,
                kernel_size=kernel_size,
                stride=2,
                drop_rate=drop_rate
            ),

            # Block 5: 128 -> 128, stride 1
            ResidualBlock(
                in_channels=filter_length * 2,
                out_channels=filter_length * 2,
                kernel_size=kernel_size,
                stride=1,
                drop_rate=drop_rate
            ),

            # Block 6: 128 -> 128, stride 2
            ResidualBlock(
                in_channels=filter_length * 2,
                out_channels=filter_length * 2,
                kernel_size=kernel_size,
                stride=2,
                drop_rate=drop_rate
            ),

            # Block 7: 128 -> 128, stride 1
            ResidualBlock(
                in_channels=filter_length * 2,
                out_channels=filter_length * 2,
                kernel_size=kernel_size,
                stride=1,
                drop_rate=drop_rate
            ),

            # Block 8: 128 -> 256, stride 2
            ResidualBlock(
                in_channels=filter_length * 2,
                out_channels=filter_length * 4,
                kernel_size=kernel_size,
                stride=2,
                drop_rate=drop_rate
            ),

            # Block 9: 256 -> 256, stride 1
            ResidualBlock(
                in_channels=filter_length * 4,
                out_channels=filter_length * 4,
                kernel_size=kernel_size,
                stride=1,
                drop_rate=drop_rate
            ),

            # Block 10: 256 -> 256, stride 2
            ResidualBlock(
                in_channels=filter_length * 4,
                out_channels=filter_length * 4,
                kernel_size=kernel_size,
                stride=2,
                drop_rate=drop_rate
            ),

            # Block 11: 256 -> 256, stride 1
            ResidualBlock(
                in_channels=filter_length * 4,
                out_channels=filter_length * 4,
                kernel_size=kernel_size,
                stride=1,
                drop_rate=drop_rate
            ),

            # Block 12: 256 -> 512, stride 2
            ResidualBlock(
                in_channels=filter_length * 4,
                out_channels=filter_length * 8,
                kernel_size=kernel_size,
                stride=2,
                drop_rate=drop_rate
            ),

            # Block 13: 512 -> 512, stride 1
            ResidualBlock(
                in_channels=filter_length * 8,
                out_channels=filter_length * 8,
                kernel_size=kernel_size,
                stride=1,
                drop_rate=drop_rate
            ),

            # Block 14: 512 -> 512, stride 2
            ResidualBlock(
                in_channels=filter_length * 8,
                out_channels=filter_length * 8,
                kernel_size=kernel_size,
                stride=2,
                drop_rate=drop_rate
            ),
        )

        # ============================================================
        # Block 3: Output block
        # ============================================================
        self.output_block = nn.Sequential(
            nn.BatchNorm1d(filter_length * 8),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate final flattened size
        final_length = input_size

        # 8 blocks have stride=2
        for _ in range(8):
            final_length = final_length // 2

        self.classifier = nn.Linear(
            filter_length * 8 * final_length,
            num_classes
        )

        self._initialize_weights()

    def forward(self, x):

        # Keras: (batch, length, channels)
        # PyTorch: (batch, channels, length)
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.transpose(1, 2)

        # ------------------------------------------------------------
        # Block 1
        # ------------------------------------------------------------
        shortcut = x
        x = self.first_block(x)

        # pool_size=1, stride=1 is an identity operation
        x = shortcut + x

        # ------------------------------------------------------------
        # Block 2
        # ------------------------------------------------------------
        x = self.main_blocks(x)

        # ------------------------------------------------------------
        # Block 3
        # ------------------------------------------------------------
        x = self.output_block(x)

        x = self.classifier(x)

        return x


class ResidualBlock(nn.Module):
    """
    One residual block used inside main_blocks.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        drop_rate
    ):
        super(ResidualBlock, self).__init__()

        padding = kernel_size // 2

        self.bn1 = nn.BatchNorm1d(in_channels)

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(drop_rate)

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )

        # Shortcut
        self.pool = nn.MaxPool1d(
            kernel_size=stride,
            stride=stride
        )

        if in_channels != out_channels:
            self.channel_pad = nn.ZeroPad2d(
                (0, 0, 0, out_channels - in_channels)
            )
        else:
            self.channel_pad = nn.Identity()

    def forward(self, x):

        # Shortcut
        shortcut = self.pool(x)

        if isinstance(self.channel_pad, nn.ZeroPad2d):
            shortcut = self.channel_pad(shortcut)

        # Main branch
        out = self.bn1(x)
        out = F.relu(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out)

        out = self.dropout(out)

        out = self.conv2(out)

        # Residual addition
        return shortcut + out

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
    
    
def evaluate_and_plot_cm(model, dataloader, device, name):
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
    print(f"\n--- Classification Report {name} dataset ---")
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
    plt.title(f'MIT-BIH {name} Classification Confusion Matrix')
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