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
print(f"Python version: {torch.sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available? {torch.cuda.is_available()}")

directory = lib.directory
CLASS_NAMES = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']
# Mapping MIT-BIH symbols to standard 5 AAMI classes
AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal / Non-ectopic
    'A': 1, 'a': 1, 'J': 1, 'S': 1,          # Supraventricular Ectopic
    'V': 2, 'E': 2,                          # Ventricular Ectopic
    'F': 3,                                  # Fusion Beat
    '/': 4, 'f': 4, 'Q': 4                   # Unknown / Paced / Artifacts
}

def plot_beats_per_subject(record_list=None):
    if record_list is None:
        record_list = lib.train_val_records + lib.test_records

    beats_per_subject = Counter()
    for record_name in record_list:
        annotation = wfdb.rdann(directory + record_name, 'atr')
        subject = '201/202' if record_name in ('201', '202') else record_name
        beats_per_subject[subject] += sum(
            symbol in AAMI_MAPPING for symbol in annotation.symbol
        )

    subjects = sorted(
        beats_per_subject,
        key=lambda subject: 201 if subject == '201/202' else int(subject)
    )
    counts = [beats_per_subject[subject] for subject in subjects]

    plt.figure(figsize=(12, 6))
    plt.bar(subjects, counts, color='skyblue', edgecolor='black')
    plt.title('Number of beats per subject')
    plt.xlabel('Subject')
    plt.ylabel('Number of beats')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return beats_per_subject

def plot_signal_amplitude_distribution(record_list=None, channel=0, bins=100):
    if record_list is None:
        record_list = lib.train_val_records + lib.test_records
    if not record_list:
        raise ValueError("record_list must contain at least one record")

    amplitude_min = float('inf')
    amplitude_max = float('-inf')
    for record_name in record_list:
        record = wfdb.rdrecord(directory + record_name, channels=[channel])
        signal = record.p_signal[:, 0]
        amplitude_min = min(amplitude_min, float(np.min(signal)))
        amplitude_max = max(amplitude_max, float(np.max(signal)))
    amplitude_min=-3
    amplitude_max=3
    bin_edges = np.linspace(amplitude_min, amplitude_max, bins + 1)
    counts = np.zeros(bins, dtype=np.int64)
    for record_name in record_list:
        record = wfdb.rdrecord(directory + record_name, channels=[channel])
        signal = record.p_signal[:, 0]
        counts += np.histogram(signal, bins=bin_edges)[0]
    print(amplitude_min)
    print(amplitude_max)
    plt.figure(figsize=(10, 6))
    plt.stairs(counts, bin_edges, fill=True, color='skyblue', edgecolor='black')
    plt.title('ECG signal amplitude distribution')
    plt.xlabel('Signal amplitude (mV)')
    plt.ylabel('Number of samples')
    plt.tight_layout()
    plt.show()
    return counts, bin_edges

def find_and_plot_noisy_beat(record_list=None, channel=0, window_size=100):
    if record_list is None:
        record_list = lib.train_val_records + lib.test_records
    if not record_list:
        raise ValueError("record_list must contain at least one record")
    if window_size < 5:
        raise ValueError("window_size must be at least 5 samples")

    best_beat = None
    best_info = None
    best_noise_score = -1.0
    half_window = window_size // 2
    smoothing_kernel = np.ones(5) / 5

    for record_name in record_list:
        record = wfdb.rdrecord(directory + record_name, channels=[channel])
        signal = record.p_signal[:, 0]
        annotation = wfdb.rdann(directory + record_name, 'atr')

        for sample_idx, symbol in zip(annotation.sample, annotation.symbol):
            if symbol not in AAMI_MAPPING:
                continue

            start_idx = sample_idx - half_window
            end_idx = start_idx + window_size
            if start_idx < 0 or end_idx > len(signal):
                continue

            beat = signal[start_idx:end_idx]
            padded_beat = np.pad(beat, (2, 2), mode='edge')
            smoothed_beat = np.convolve(padded_beat, smoothing_kernel, mode='valid')
            noise_score = float(np.sqrt(np.mean((beat - smoothed_beat) ** 2)))
            if noise_score > best_noise_score:
                best_beat = beat.copy()
                best_info = (record_name, int(sample_idx), symbol)
                best_noise_score = noise_score

    if best_beat is None:
        raise ValueError("No valid annotated beats found in the selected records")

    time_ms = (np.arange(window_size) - half_window) / 360 * 1000
    smoothed_beat = np.convolve(
        np.pad(best_beat, (2, 2), mode='edge'),
        smoothing_kernel,
        mode='valid'
    )
    plt.figure(figsize=(10, 5))
    plt.plot(time_ms, best_beat, label='ECG beat')
    plt.plot(time_ms, smoothed_beat, label='Smoothed signal', linestyle='--')
    plt.title(
        f'Noisiest beat: record {best_info[0]}, sample {best_info[1]} '
        f'(symbol {best_info[2]}, noise score {best_noise_score:.4f})'
    )
    plt.xlabel('Time relative to annotated beat (ms)')
    plt.ylabel('Signal amplitude (mV)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return best_info, best_noise_score

#
# print("Train")
#print(train_val_records[:int(train_val_split*len(train_val_records))])

# 2. Instantiate PyTorch Datasets
#train_dataset = MITBIHDataset(record_list=train_val_records[:int(train_val_split*len(train_val_records))], window_size=256)
#val_dataset   = MITBIHDataset(record_list=train_val_records[int(train_val_split*len(train_val_records)):], window_size=256)
#test_dataset   = MITBIHDataset(record_list=test_records, window_size=256)
full_dataset = lib.MITBIHDataset(record_list=lib.train_val_records+lib.test_records, window_size=100)

# 3. Create PyTorch DataLoaders
#train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)
#test_loader   = DataLoader(test_dataset, batch_size=64, shuffle=False)

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

log_plot_balance(full_dataset)
plot_beats_per_subject()
plot_signal_amplitude_distribution()
find_and_plot_noisy_beat()


def print_plot_balance(dataset):
    counts = Counter(dataset.labels)
    #print(counts[0])
    counts = [counts[i] for i in range(len(CLASS_NAMES))]
    #print(counts[0])
    count_sum = sum(counts)
    for i in range(len(counts)):
        counts[i]=counts[i]*100/count_sum
    print(counts)


for i in range(len(lib.train_val_records)):
    temp = lib.MITBIHDataset(record_list=[lib.train_val_records[i]], window_size=256)
    print_plot_balance(temp)

name = '105'
record = wfdb.rdrecord(lib.directory+name, sampto=3600)  # First 10 seconds (360 Hz * 10s)
annotation = wfdb.rdann(lib.directory+name, 'atr', sampto=3600)

# 2. Plot signals with overlaid beat markers
wfdb.plot_wfdb(
    record=record, 
    annotation=annotation,
    plot_sym=True,
    title="MIT-BIH Record "+name+" (Lead II & V1)",
    time_units="seconds",
    figsize=(12, 6)
)