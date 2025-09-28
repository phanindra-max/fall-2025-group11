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
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT) # loads pretrained weights from ImageNet
        self.backbone.fc = nn.Identity() # Removes the final classification layer so the network returns features, not class predictions.
        self.out_dim = 512 # Output size from ResNet18’s penultimate layer.

    def forward(self, x):
        return self.backbone(x) # Pass input x through the modified backbone to get 512-dim feature vectors.


class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes): # in_dim = 512 from above feature exractor
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes) # linear classifier that maps features to logits

    def forward(self, x): # returns raw prediction logits
        return self.fc(x)


# ---------------- Losses ----------------

class TargetMarginLoss(nn.Module):
    def __init__(self, margin=0.5, scale=30):
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, features, labels, weights): # features = normalised output from feature extractor, weights  = normalised learnable class weights,
        # features = tensor of shape[batch_size, feature_dim]
        # labels = tensor of shape[batch_size]
        # weights = tensor of shape[num_classes, feature_dim]
        cos_theta = F.linear(F.normalize(features), F.normalize(weights)) # cosine similarity between feature and class weights
        target_logit = cos_theta[range(len(labels)), labels] # correct label based on cos theta values is chosen for each sample
        margin_theta = torch.acos(torch.clamp(target_logit, -1.0 + 1e-7, 1.0 - 1e-7)) + self.margin # torch.acos converts cosine similarity to angle in radians.clamp allows to change the input to acos to range [-1,1]; which means any value lesser or grater than this range will be changed to these values. self.margin allows to add a margin to this angle
        cos_theta[range(len(labels)), labels] = torch.cos(margin_theta) # replacing original cosine value of class with new lower value
        return self.scale * cos_theta # scaled cosine logits to stabilize training


def entropy_loss(logits): # take logits - raw outputs from classifier (before softmax)
    probs = F.softmax(logits, dim=1) # converting logits to probabilities using softmax, probs of size [batch_size, num_classes]
    return -torch.mean(torch.sum(probs * torch.log(probs + 1e-6), dim=1)) # calculate probab


# ---------------- Q-Learning Agent ----------------

class QNetwork(nn.Module): # initialising deep Q network
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x)) # 2 hidden layers with ReLU activation function
        x = F.relu(self.fc2(x))
        return self.out(x) # output Q-values for each action. Q values represent expected reward for each action taken


class ReplayBuffer: # just like a memory bank that stores experiences(state, action, reward, next state) and replays them in small batches while updating the network/model instead of updating the network for every new experience.
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity) # buffer that holds experiences

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state)) # add new experience

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size) # randomly select batch of experiences/transitions for training

    def __len__(self):
        return len(self.buffer) # current size of buffer


# ---------------- Utility Functions ----------------

def compute_reward(confidence, pf, delta_e, beta=0.2, lam=0.3):
    phi = confidence - beta * pf - lam * delta_e # ϕ = confidence − β × frequency_penalty − λ × entropy_penalty i.e. reward = high confidence - overuse penalty - uncertainity penalty. this essentially ensures RL agent to favor : confident samples, less frequently chosen samples, samples with decreasing entropy
    return 1.0 if phi >= 0.5 else -1.0


def compute_entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-8)) # calculating entropy of probability vector probs


def generate_pseudo_labels(model, data_loader, device):
    model.eval() # Turns off dropout and batch normalization layers, Ensures the model behaves deterministically for inference.
    pseudo_labels = [] # predicted class
    all_features = [] # raw feature embeddings from feature extractor
    all_probs = [] # soft max probabilities

    with torch.no_grad(): # disable gradient tracking to save memory and increase speed
        for x, _ in data_loader: # batches of unlabeled data is considered
            x = x.to(device)
            features = model.feature_extractor(x) # 512 dim features extraction
            logits = model.classifier(features) # feed features to classifer
            probs = F.softmax(logits, dim=1) # apply softmax to logits to get probabilities (one vector per input image)
            preds = torch.argmax(probs, dim=1) # class with highest softmax probability is chosen as predicted class
            # saving results in cpu and numpy format
            pseudo_labels.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_features.extend(features.cpu().numpy())

    return pseudo_labels, np.array(all_probs), np.array(all_features) # returned for later reward calculation


def compute_class_centers(features, labels, num_classes): # computing mean feature vector/centroid for each class
    centers = np.zeros((num_classes, features.shape[1]))
    counts = np.zeros(num_classes) # number of samples each class has
    for f, y in zip(features, labels): #
        centers[y] += f # adding feature vector to its class accumulator
        counts[y] += 1 # incrementing class counter as well
    for i in range(num_classes):
        if counts[i] > 0:
            centers[i] /= counts[i] # averaging accumulated features to get centers
    return centers # dim: (num_classes, feature_dim)


def compute_pf(xi, yi, centers):
    dist = np.linalg.norm(xi - centers[yi]) # euclidean distance is calculated between feature vector xi and center of its predicted class)
    return dist # this is used to penalize samples far from its class center - reward function


def construct_state(xi, pi, D_t_features, D_u_features): # state representation for RL
    return np.concatenate([xi, pi, np.mean(D_t_features, axis=0), np.mean(D_u_features, axis=0)]) # xi=feature vector, pi=softmax probability vector, D_t_features - features from pseudo labeled data, D_u_features - features from unlabeled data. concatenates all to get a state vector(input to Q network to decide to pseudo label or not)


def evaluate(model, data_loader, device):
    model.eval() # disable gradients
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device, dtype=torch.long)
            logits = model(x) # get logits
            preds = torch.argmax(logits, dim=1) # get predictions using arg max
            correct += (preds == y).sum().item() # compares with ground truth y and counts correct predictions
            total += y.size(0) # total predictions
    return correct / total # accuracy = correct/total predictions


# ---------------- Model ----------------

class CombinedModel(nn.Module): # wraps both classifier and feature extractor
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = FeatureExtractor() # resnet18 with last FC removed
        self.classifier = Classifier(self.feature_extractor.out_dim, num_classes) # A linear layer mapping 512-dim features → num_classes logits.

    def forward(self, x): # input → feature extractor → classifier → logits.
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
