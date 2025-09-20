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
from torch.utils.data import DataLoader
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

# -------------------------------
# 6. Simple RL Env skeleton (accept/skip)
# -------------------------------
class PseudoLabelEnv:
    """Minimal skeleton for RL agent: accept/skip"""
    def __init__(self, unlabeled_data, model=None):
        self.unlabeled = unlabeled_data  # list of (x, idx, true_label)
        self.model = model
        self.ptr = 0
        self.done = False
        
    def reset(self):
        self.ptr = 0
        self.done = False
        return self._get_obs()
    
    def _get_obs(self):
        x, idx, y = self.unlabeled[self.ptr]
        if self.model is None:
            probs = np.ones(10)/10
        else:
            self.model.eval()
            with torch.no_grad():
                tensor = torch.tensor(x).unsqueeze(0).float()
                logits, _ = self.model(tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy().squeeze()
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        obs = np.concatenate([probs, np.array([entropy])])
        return obs
    
    def step(self, action):
        """Action: 0=skip, 1=accept; reward=+1 if correct else -1"""
        x, idx, true_label = self.unlabeled[self.ptr]
        obs = self._get_obs()
        reward = 0
        if action == 1 and self.model is not None:
            with torch.no_grad():
                logits, _ = self.model(torch.tensor(x).unsqueeze(0).float())
                pred = torch.argmax(logits, dim=1).item()
                reward = 1 if pred==true_label else -1
        self.ptr +=1
        if self.ptr>=len(self.unlabeled):
            self.done = True
        next_obs = self._get_obs() if not self.done else np.zeros_like(obs)
        return next_obs, reward, self.done

# -------------------------------
# 7. Example of setting up MNIST loaders
# -------------------------------
transform = transforms.ToTensor()
mnist_train = datasets.MNIST('.', train=True, download=True, transform=transform)
labeled_subset = torch.utils.data.Subset(mnist_train, list(range(1000)))
unlabeled_subset = [(mnist_train[i][0].numpy(), i, mnist_train[i][1]) for i in range(1000,2000)]
train_loader = DataLoader(labeled_subset, batch_size=64, shuffle=True)

# -------------------------------
# 8. Quick run demo
# -------------------------------
device = torch.device('cpu')
model = DownstreamModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_warmup(model, device, train_loader, optimizer, epochs=1)
soft_labels = generate_soft_labels(model, unlabeled_subset, device)
print("Soft pseudo-labels generated for unlabeled subset:", soft_labels.shape)
