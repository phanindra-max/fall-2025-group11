# selective_pseudo_labeling.py
# I have referred this code from internet for understanding the research paper better by tweaking it

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from collections import deque
import numpy as np
import random


# ---------------- Base Model ----------------

class FeatureExtractor(nn.Module):
    def __init__(self, base_model='resnet18'):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.out_dim = 512

    def forward(self, x):
        return self.backbone(x)


class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# ---------------- Losses ----------------

class TargetMarginLoss(nn.Module):
    def __init__(self, margin=0.5, scale=30):
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, features, labels, weights):
        cos_theta = F.linear(F.normalize(features), F.normalize(weights))
        target_logit = cos_theta[range(len(labels)), labels]
        margin_theta = torch.acos(torch.clamp(target_logit, -1.0 + 1e-7, 1.0 - 1e-7)) + self.margin
        cos_theta[range(len(labels)), labels] = torch.cos(margin_theta)
        return self.scale * cos_theta


def entropy_loss(logits):
    probs = F.softmax(logits, dim=1)
    return -torch.mean(torch.sum(probs * torch.log(probs + 1e-6), dim=1))


# ---------------- Q-Learning Agent ----------------

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ---------------- Utility Functions ----------------

def compute_reward(confidence, pf, delta_e, beta=0.2, lam=0.3):
    phi = confidence - beta * pf - lam * delta_e
    return 1.0 if phi >= 0.5 else -1.0


def compute_entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-8))


def generate_pseudo_labels(model, data_loader, device):
    model.eval()
    pseudo_labels = []
    all_features = []
    all_probs = []

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            features = model.feature_extractor(x)
            logits = model.classifier(features)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            pseudo_labels.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_features.extend(features.cpu().numpy())

    return pseudo_labels, np.array(all_probs), np.array(all_features)


def compute_class_centers(features, labels, num_classes):
    centers = np.zeros((num_classes, features.shape[1]))
    counts = np.zeros(num_classes)
    for f, y in zip(features, labels):
        centers[y] += f
        counts[y] += 1
    for i in range(num_classes):
        if counts[i] > 0:
            centers[i] /= counts[i]
    return centers


def compute_pf(xi, yi, centers):
    dist = np.linalg.norm(xi - centers[yi])
    return dist


def construct_state(xi, pi, D_t_features, D_u_features):
    return np.concatenate([xi, pi, np.mean(D_t_features, axis=0), np.mean(D_u_features, axis=0)])


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device, dtype=torch.long)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


# ---------------- Model ----------------

class CombinedModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = Classifier(self.feature_extractor.out_dim, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


# ---------------- Main Loop ----------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    source_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    target_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)

    # source_loader = DataLoader(source_dataset, batch_size=64, shuffle=True)
    # target_loader = DataLoader(target_dataset, batch_size=64, shuffle=False)

    # --- Mini Run: only take 10% of the data ---
    mini_size_source = int(0.1 * len(source_dataset))  # 5,000 instead of 50,000
    mini_size_target = int(0.1 * len(target_dataset))  # 1,000 instead of 10,000

    source_subset = Subset(source_dataset, range(mini_size_source))
    target_subset = Subset(target_dataset, range(mini_size_target))

    source_loader = DataLoader(source_subset, batch_size=64, shuffle=True)
    target_loader = DataLoader(target_subset, batch_size=64, shuffle=False)

    model = CombinedModel(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # -------- Baseline: Pretrain on source only --------
    model.train()
    for epoch in range(2):
        for x, y in source_loader:
            x, y = x.to(device),  y.to(device, dtype=torch.long)
            features = model.feature_extractor(x)
            logits = model.classifier(features)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    baseline_acc = evaluate(model, target_loader, device)
    print(f"🔹 Baseline (source only) accuracy: {baseline_acc * 100:.2f}%")

    # -------- RL-based pseudo-labeling --------
    pseudo_labels, pseudo_probs, pseudo_features = generate_pseudo_labels(model, target_loader, device)
    centers = compute_class_centers(pseudo_features, pseudo_labels, num_classes=10)

    qnet = QNetwork(state_dim=512 + 10 + 512*2, action_dim=10).to(device)
    q_optimizer = optim.Adam(qnet.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer()

    D_t_features = pseudo_features[:100]
    D_u_features = pseudo_features[100:]
    D_u_probs = pseudo_probs[100:]
    D_u_labels = pseudo_labels[100:]

    for i in range(len(D_u_features)):
        xi = D_u_features[i]
        pi = D_u_probs[i]
        yi = D_u_labels[i]
        state = construct_state(xi, pi, D_t_features, D_u_features)

        pf = compute_pf(xi, yi, centers)
        entropy_before = np.mean([compute_entropy(p) for p in D_u_probs])
        D_t_features = np.vstack([D_t_features, xi])
        centers = compute_class_centers(D_t_features, np.append(np.zeros(len(D_t_features)-1, dtype=int), int(yi)), 10)
        entropy_after = np.mean([compute_entropy(p) for p in D_u_probs])
        delta_e = entropy_before - entropy_after
        reward = compute_reward(np.max(pi), pf, delta_e)

        next_state = construct_state(xi, pi, D_t_features, D_u_features)
        replay_buffer.push(torch.tensor(state, dtype=torch.float32), i % 10, reward, torch.tensor(next_state, dtype=torch.float32))

        if len(replay_buffer) > 16:
            batch = replay_buffer.sample(16)
            states, actions, rewards, next_states = zip(*batch)
            states = torch.stack(states).to(device)
            actions = torch.tensor(actions).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            next_states = torch.stack(next_states).to(device)

            q_values = qnet(states)
            next_q_values = qnet(next_states)
            target_q = rewards + 0.9 * torch.max(next_q_values, dim=1)[0]

            q_loss = F.mse_loss(q_values.gather(1, actions.unsqueeze(1)).squeeze(), target_q)
            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()

    rl_acc = evaluate(model, target_loader, device)
    print(f"✅ RL-based pseudo-labeling accuracy: {rl_acc * 100:.2f}%")


if __name__ == '__main__':
    main()
