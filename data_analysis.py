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



window_size = 64

train_dataset = lib.MITBIHDataset(record_list=lib.train_records, window_size=window_size)
val_dataset   = lib.MITBIHDataset(record_list=lib.val_records, window_size=window_size)
test_dataset   = lib.MITBIHDataset(record_list=lib.test_records, window_size=window_size)


def log_plot_balance(dataset):
    counts = Counter(dataset.labels)
    #print(counts[0])
    counts = [counts[i] for i in range(len(lib.CLASS_NAMES))]
    #print(counts[0])
    count_sum = sum(counts)
    for i in range(len(counts)):
        counts[i]=counts[i]*100/count_sum
    print(counts)
    # Skapa histogrammet
    plt.bar(lib.CLASS_NAMES,counts, color='skyblue', edgecolor='black', log=True)
    # Lägg till titlar och etiketter
    plt.title('Frequency of labels')
    plt.xlabel('Labels')
    plt.ylabel('Procent')
    plt.show()

def beat_plot(dataset,target_label):
    all_labels = torch.tensor([label for _, label in dataset])

    # 2. Find indices for your target label
    indices = (all_labels == target_label).nonzero(as_tuple=True)[0]


    for i in indices:
        signal = dataset[i][0][0]
        if(signal.max() < torch.abs(signal.min())):
            signal = -signal
        plt.plot(range(len(signal)),signal, color='blue',alpha=0.2)        


beat_plot(test_dataset,0)

name = '101'
record = wfdb.rdrecord(lib.directory+name, sampto=3600)  # First 10 seconds (360 Hz * 10s)
annotation = wfdb.rdann(lib.directory+name, 'atr', sampto=3600)



# Lägg till titlar och etiketter
plt.title('Frequency of labels')
plt.xlabel('Labels')
plt.ylabel('Procent')
plt.show()


log_plot_balance(train_dataset)

# 2. Plot signals with overlaid beat markers
wfdb.plot_wfdb(
    record=record, 
    annotation=annotation,
    plot_sym=True,
    title="MIT-BIH Record "+name+" (Lead II & V1)",
    time_units="seconds",
    figsize=(12, 6)
)    