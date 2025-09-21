# ===============================
# Minimal Base for RL Pseudo-Labeling Project
# ===============================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy
print(torch.__version__, torchvision.__version__, numpy.__version__)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# -------------------------------
# 1. Downstream Model
# -------------------------------
class DownstreamModel(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, embedding_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*14*14 if input_channels==1 else 64*8*8, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        emb = F.relu(self.fc1(x))
        logits = self.fc2(emb)
        return logits, emb

# -------------------------------
# 2. Loss Functions
# -------------------------------
def pseudo_label_loss(pred_logits, soft_targets):
    """KL divergence between predicted logits and soft labels"""
    log_probs = F.log_softmax(pred_logits, dim=1)
    loss = F.kl_div(log_probs, soft_targets, reduction='batchmean')
    return loss

def ra_rh_regularizers(all_logits, lambda_ra=0.1, lambda_rh=0.1):
    """RA: class-balance, RH: low entropy"""
    probs = F.softmax(all_logits, dim=1)
    # RA: encourage uniform distribution
    avg_probs = torch.mean(probs, dim=0) + 1e-8
    uniform = torch.ones_like(avg_probs) / avg_probs.size(0)
    ra_loss = F.kl_div(torch.log(avg_probs), uniform, reduction='sum')
    # RH: encourage sharp predictions
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    rh_loss = torch.mean(entropy)
    return lambda_ra*ra_loss, lambda_rh*rh_loss

# -------------------------------
# 3. Mixup augmentation
# -------------------------------
def mixup_data(x, y, alpha=1.0):
    """Simple mixup"""
    if alpha <= 0: return x, y
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y

# -------------------------------
# 4. Warm-up training
# -------------------------------
def train_warmup(model, device, train_loader, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Warmup epoch {epoch+1}/{epochs} done.")

# -------------------------------
# 5. Generate soft pseudo-labels
# -------------------------------
def generate_soft_labels(model, dataset, device):
    model.eval()
    soft_labels = []
    with torch.no_grad():
        for item in dataset:
            if len(item) == 3:
                x, idx, y = item
            else:
                x, y = item
            # Convert numpy array to tensor if needed
            if isinstance(x, np.ndarray):
                x = torch.tensor(x).float()
            x = x.unsqueeze(0).to(device)
            logits, _ = model(x)
            probs = F.softmax(logits, dim=1)
            soft_labels.append(probs.squeeze(0).cpu())
    return torch.stack(soft_labels)

# Confidence-based selection
# -----------------------
high_conf_mask = torch.max(soft_labels, dim=1).values >= confidence_threshold
selected_images = torch.stack([img for i, img in enumerate(unlabeled_images) if high_conf_mask[i]])
selected_labels = torch.stack([soft_labels[i] for i in range(len(soft_labels)) if high_conf_mask[i]])

print(f"Selected {len(selected_images)} high-confidence pseudo-labels")

# Convert soft labels to class indices for training
pseudo_targets = torch.argmax(selected_labels, dim=1)

# -----------------------
#  Create downstream dataset (labeled + pseudo-labeled)
# -----------------------
# Convert labeled subset to tensors
labeled_x = torch.stack([x for x, y in labeled_subset])
labeled_y = torch.tensor([y for x, y in labeled_subset])

# Combine
train_x = torch.cat([labeled_x, selected_images])
train_y = torch.cat([labeled_y, pseudo_targets])

train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# -----------------------
#  Mixup augmentation function
# -----------------------
def mixup_data(x, y, alpha=mixup_alpha):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# -----------------------
#  Train downstream model with mixup
# -----------------------
downstream_model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(downstream_model.parameters(), lr=1e-3)

downstream_epochs = 3
for epoch in range(downstream_epochs):
    downstream_model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x, y_a, y_b, lam = mixup_data(x, y)
        optimizer.zero_grad()
        logits = downstream_model(x)
        loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
        loss.backward()
        optimizer.step()
print("Downstream training with mixup done!")

# -----------------------
# Evaluate downstream model
# -----------------------
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

downstream_model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = downstream_model(x)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

print("Downstream model accuracy:", correct/total)