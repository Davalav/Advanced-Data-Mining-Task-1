import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
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
from ecgdetectors import Detectors
import time
start_time=time.time()
time_a= time.time()
window_size = 64
offset = 1
train_dataset = lib.MITBIHDataset(record_list=lib.train_records, window_size=window_size, channel=0, offset=offset)
val_dataset   = lib.MITBIHDataset(record_list=lib.val_records, window_size=window_size, channel=0, offset=offset)
test_dataset   = lib.MITBIHDataset(record_list=lib.test_records, window_size=window_size, channel=0, offset=offset)
elapsed_time=time.time()-time_a
print(f"Dataset init took {elapsed_time} seconds")
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
    plt.ylabel('Frequency percentage')
    plt.show()

def beat_plot(dataset,target_label,name, max_length=-1):
    if max_length==-1:
        max_length=len(indices)
    all_labels = torch.tensor([label for _, label in dataset])

    # 2. Find indices for your target label
    indices = (all_labels == target_label).nonzero(as_tuple=True)[0]
    print(f"Samples: {len(indices)}, max_length: {max_length}")
    for i in indices[:max_length]:
        signal = dataset[i][0][0]
        #if(signal.max() < torch.abs(signal.min())):
            #signal = -signal
        plt.plot(range(len(signal)),signal, color='blue',alpha=max(20/min(len(indices),max_length),0.01))  
    # Lägg till titlar och etiketter
    plt.title(f'Beat overlap from: {name} dataset, {lib.CLASS_NAMES[target_label]}')
    plt.xlabel('Time [ms]')
    plt.ylabel('MLIImV')
    plt.show()

def compared_plot(dataset,name, max_length=-1):
    if max_length==-1:
        max_length=len(indices)
    all_labels = torch.tensor([label for _, label in dataset])
    colours = ['blue', 'red', 'green', 'yellow']
    for target_label in range(4):
        # 2. Find indices for your target label
        indices = (all_labels == target_label).nonzero(as_tuple=True)[0]
        print(f"Samples: {len(indices)}, max_length: {max_length}")
        for i in indices[:max_length]:
            signal = dataset[i][0][0]
            #if(signal.max() < torch.abs(signal.min())):
                #signal = -signal
            plt.plot(range(len(signal)),signal, color=colours[target_label],alpha=max(20/min(len(indices),max_length),0.01))  
    # Lägg till titlar och etiketter
    plt.title(f'Beat overlap comparison from: {name} dataset')
    plt.xlabel('Time [ms]')
    plt.ylabel('MLIImV')
    plt.show()
time_a= time.time()


#beat_plot(train_dataset,0,"Training",1000)
#beat_plot(train_dataset,1,"Training",1000)
#beat_plot(train_dataset,2,"Training",1000)
#beat_plot(train_dataset,3,"Training",1000)

#compared_plot(train_dataset,"Training", 100)


#log_plot_balance(train_dataset)
elapsed_time=time.time()-time_a
print(f"Individual beat plots took {elapsed_time} seconds")
name = '101'
record = wfdb.rdrecord(lib.directory+name, sampto=3600)  # First 10 seconds (360 Hz * 10s)
annotation = wfdb.rdann(lib.directory+name, 'atr', sampto=3600)
detectors = Detectors(360)

signal = record.p_signal[:, 0]
r_peaks = detectors.pan_tompkins_detector(signal)
extra_symbols = ['x'] * len(r_peaks) 

# 3. Slå ihop samplenumren och symbolerna
combined_samples = np.concatenate((annotation.sample, r_peaks))
combined_symbols = np.array(annotation.symbol + extra_symbols)

# 4. VIKTIGT: Sortera efter samplenummer (wfdb kräver kronologisk ordning)
sort_indices = np.argsort(combined_samples)
sorted_samples = combined_samples[sort_indices]
sorted_symbols = combined_symbols[sort_indices].tolist()  # Konvertera tillbaka till lista


ann_chans = np.zeros(len(sorted_samples), dtype=int)

# 5. Skapa det nya kombinerade Annotation-objektet
combined_ann = wfdb.Annotation(
    record_name=record.record_name,
    extension='combined',
    sample=sorted_samples,
    symbol=sorted_symbols,
    chan=ann_chans  
)

# 2. Plot signals with overlaid beat markers
wfdb.plot_wfdb(
    record=record, 
    annotation=combined_ann,
    plot_sym=True,
    title="MIT-BIH Record "+name+" (Lead II & V1)",
    time_units="seconds",
    figsize=(12, 6)
)

print(f"Total time: {time.time()-start_time} seconds")