import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
