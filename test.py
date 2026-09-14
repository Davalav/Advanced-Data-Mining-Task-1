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



#print("Train")
#print(lib.train_val_records[:int(lib.train_val_split*len(lib.train_val_records))])
#print("Val")
#print(lib.train_val_records[int(lib.train_val_split*len(lib.train_val_records)):])

# 2. Instantiate PyTorch Datasets
train_dataset = lib.MITBIHDataset(record_list=lib.train_records, window_size=256)
val_dataset   = lib.MITBIHDataset(record_list=lib.val_records, window_size=256)
test_dataset   = lib.MITBIHDataset(record_list=lib.test_records, window_size=256)

# 3. Create PyTorch DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader   = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 4. Inspect batch shapes inside training loop
for x_batch, y_batch in train_loader:
    print(f"Input batch shape (Batch, Channels, Window): {x_batch.shape}")  # e.g., torch.Size([64, 1, 256])
    print(f"Target batch shape (Batch): {y_batch.shape}")                   # e.g., torch.Size([64])
    break





lib.log_plot_balance(train_dataset)
lib.log_plot_balance(val_dataset)
lib.log_plot_balance(test_dataset)




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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Initialize Model, Loss, and Optimizer

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model = lib.ECG1DCNN(num_classes=5).to(device)
model.apply(init_weights)

class_weights = lib.compute_class_weights(train_dataset.labels, num_classes=5)
class_weights = class_weights.to(device)
print("Computed Class Weights:")
for i, w in enumerate(class_weights):
    print(f"  Class {i}: {w.item():.4f}")

# 3. Pass the weights directly into CrossEntropyLoss
#criterion = nn.CrossEntropyLoss(weight=class_weights)
#criterion = nn.CrossEntropyLoss()

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weight tensor
        self.gamma = gamma  # Focusing parameter (typically 2.0)

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class StableFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Numerically stable Focal Loss using log_softmax.
        
        Args:
            alpha (Tensor, optional): Class weights tensor of shape (num_classes,).
            gamma (float): Focusing parameter (default 2.0).
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(StableFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. Compute log probabilities safely
        log_pt = F.log_softmax(inputs, dim=1)
        
        # 2. Extract log_pt for true class targets
        log_pt = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()  # Convert back to probability safely

        # 3. Compute Focal Loss formula: -alpha * (1 - pt)^gamma * log(pt)
        focal_weight = (1.0 - pt) ** self.gamma
        loss = -focal_weight * log_pt

        # 4. Apply optional class alpha weights
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets)
            loss = loss * at

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# Usage
class_weights = class_weights/class_weights.mean()
criterion = StableFocalLoss(alpha=class_weights, gamma=1.0)

optimizer = optim.Adam(model.parameters(), lr=1e-5)
lib.count_parameters(model)
# Run Training Loop
num_epochs = 20
best_val_loss= 100
val_loss_list = []
val_acc_list=[]
train_loss_list= []

best_model = copy.deepcopy(model)


for epoch in range(num_epochs):
    epoch_start_time = time.time()
    train_loss, train_acc = lib.train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc   = lib.evaluate(model, val_loader, criterion, device)
    val_acc_list.append(val_acc)
    val_loss_list.append(val_loss)
    train_loss_list.append(train_loss)
    if(val_loss< best_val_loss):
        print("New best found!")
        best_val_loss= val_loss
        best_model = copy.deepcopy(model)
    print(f"Epoch {epoch:02d}/{num_epochs:02d} | "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc * 100:.2f}%")
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch execution time: {lib.format_seconds(epoch_time)}")
    num_epochs_left = num_epochs - epoch - 1
    est_time_remaining = epoch_time * num_epochs_left
    print(f"Estimated time remaining: {lib.format_seconds(est_time_remaining)}")
    
model = copy.deepcopy(best_model.to(device))
current_time = time.strftime("%Y-%m-%d %H_%M_%S")
test_loss, test_acc   = lib.evaluate(model, test_loader, criterion, device)
torch.save(model.state_dict(), f"models/{model.__class__.__name__}-{val_acc:.4f}-{test_acc:.4f}_{current_time}.pth")

print(f"Final test | "
            f"Test Loss: {test_loss:.4f} - Test Acc: {test_acc * 100:.2f}%")

cm =lib.evaluate_and_plot_cm(model,test_loader,device)

plt.plot(range(1,num_epochs+1),train_loss_list)
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
    
plt.plot(range(1,num_epochs+1),val_loss_list)
plt.title('Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.plot(range(1,num_epochs+1),val_acc_list)
plt.title('Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()