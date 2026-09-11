import torch
print(f"Python version: {torch.sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available? {torch.cuda.is_available()}")

import numpy as np
from torch.utils.data import Dataset, DataLoader
import wfdb

# Mapping MIT-BIH symbols to standard 5 AAMI classes
AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal / Non-ectopic
    'A': 1, 'a': 1, 'J': 1, 'S': 1,          # Supraventricular Ectopic
    'V': 2, 'E': 2,                          # Ventricular Ectopic
    'F': 3,                                  # Fusion Beat
    '/': 4, 'f': 4, 'Q': 4                   # Unknown / Paced / Artifacts
}
directory = "../mit-bih-arrhythmia-database-1.0.0/"
train_val_split=0.7
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
                    continue
                
                start_idx = sample_idx - half_window
                end_idx = sample_idx + half_window
                
                # Boundary check
                if start_idx >= 0 and end_idx < len(signal):
                    segment = signal[start_idx:end_idx]
                    
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
    
# 1. Define train and validation record splits
train_val_records = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230']
test_records   = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']

print("Train")
print(train_val_records[:int(train_val_split*len(train_val_records))])
print("Val")
print(train_val_records[int(train_val_split*len(train_val_records)):])
# 2. Instantiate PyTorch Datasets
train_dataset = MITBIHDataset(record_list=train_val_records[:int(train_val_split*len(train_val_records))], window_size=256)
val_dataset   = MITBIHDataset(record_list=train_val_records[int(train_val_split*len(train_val_records)):], window_size=256)
test_dataset   = MITBIHDataset(record_list=test_records, window_size=256)

# 3. Create PyTorch DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader   = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 4. Inspect batch shapes inside training loop
for x_batch, y_batch in train_loader:
    print(f"Input batch shape (Batch, Channels, Window): {x_batch.shape}")  # e.g., torch.Size([64, 1, 256])
    print(f"Target batch shape (Batch): {y_batch.shape}")                   # e.g., torch.Size([64])
    break