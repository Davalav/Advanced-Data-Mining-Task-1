import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import lib
import sys
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
import torch.nn.utils.prune as prune
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
    model.load_state_dict(torch.load(sys.argv[1], map_location=device))
else:
    raise Exception("Missing file argument")