# pseudo_labeling_step4_rl.py
import os
import csv
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- RL Environment ----------------
class PseudoLabelEnv(gym.Env):
    def __init__(self, unlabeled_data, model, confidence_threshold=0.9, max_steps=1000):
        super(PseudoLabelEnv, self).__init__()
        self.unlabeled_data = unlabeled_data
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.max_steps = max_steps

        # observation space = [max_prob, entropy]
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        # action space: 0 = skip, 1 = assign pseudo-label
        self.action_space = spaces.Discrete(2)

        self.reset()

    def reset(self):
        self.indices = list(range(len(self.unlabeled_data)))
        np.random.shuffle(self.indices)
        self.ptr = 0
        self.cum_reward = 0.0
        return self._get_obs()

    def _get_obs(self):
        if self.ptr >= len(self.indices):
            return np.zeros(2, dtype=np.float32)
        x, _ = self.unlabeled_data[self.indices[self.ptr]]
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
        max_prob = probs.max()
        entropy = -(probs * np.log(probs + 1e-8)).sum()
        return np.array([max_prob, entropy], dtype=np.float32)

    def step(self, action):
        if self.ptr >= len(self.indices):
            return np.zeros(2, dtype=np.float32), 0.0, True, {}

        x, y_true = self.unlabeled_data[self.indices[self.ptr]]
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
        pred = probs.argmax()
        confidence = probs.max()

        reward = 0.0
        if action == 1:  # assign pseudo-label
            if confidence >= self.confidence_threshold:
                if pred == y_true:
                    reward = +1.0
                else:
                    reward = -1.0
            else:
                reward = -0.1
        else:  # skip
            reward = 0.0

        self.cum_reward += reward
        self.ptr += 1
        done = self.ptr >= min(len(self.indices), self.max_steps)
        return self._get_obs(), reward, done, {}

# ---------------- Policy Network ----------------
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, act_dim),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

# ---------------- Reinforce Training ----------------
def train_reinforce(policy, env, optimizer, episodes=200, gamma=0.99, results_path="results_step4_rl.csv"):
    policy.train()
    with open(results_path, "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["timestamp", "episode", "episode_reward"])

        for ep in range(episodes):
            obs = env.reset()
            log_probs = []
            rewards = []
            done = False
            steps = 0
            while not done:
                obs_v = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                logp = policy(obs_v)
                probs = logp.exp().detach().cpu().numpy().squeeze(0)
                action = np.random.choice(2, p=probs)
                log_prob_taken = logp[0, action]
                next_obs, reward, done, _ = env.step(action)
                log_probs.append(log_prob_taken)
                rewards.append(reward)
                obs = next_obs
                steps += 1
                if steps >= env.max_steps:
                    break

            # compute returns
            returns = []
            R = 0.0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32).to(device)
            if returns.std() > 0:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # policy loss
            loss = 0
            for lp, R in zip(log_probs, returns):
                loss = loss - lp * R
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log episode reward
            timestamp = datetime.datetime.now().isoformat()
            writer.writerow([timestamp, ep+1, env.cum_reward])

            if (ep+1) % max(1, episodes//10) == 0:
                print(f"EP {ep+1}/{episodes}: reward={env.cum_reward:.2f}, loss={loss.item():.4f}")

# ---------------- Example Run ----------------
if __name__ == "__main__":
    print("=== RL-BASED PSEUDO-LABELING ===")
    
    # dataset setup
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    small_subset = Subset(dataset, list(range(1000)))  # unlabeled subset
    labeled_subset = Subset(dataset, list(range(100)))  # 100 labeled samples

    # improved downstream model
    class SmallCNN(nn.Module):
        def __init__(self):
            super(SmallCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = torch.relu(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            return x

    model = SmallCNN().to(device)
    
    # STEP 1: Train initial model on labeled data
    print("Training initial model on labeled data...")
    labeled_loader = DataLoader(labeled_subset, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(10):
        for x, y in labeled_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/10, loss: {loss.item():.4f}")

    # Test initial accuracy
    test_loader = DataLoader(datasets.MNIST("./data", train=False, download=True, transform=transform),
                             batch_size=128, shuffle=False)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    initial_acc = correct / total
    print(f"Initial model accuracy: {initial_acc:.4f}")

    # STEP 2: Train RL agent
    print("\nTraining RL agent...")
    env = PseudoLabelEnv(small_subset, model, confidence_threshold=0.3)  # Much lower threshold
    policy = PolicyNetwork(obs_dim=2, act_dim=2).to(device)
    policy_optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    train_reinforce(policy, env, policy_optimizer, episodes=50)

    # STEP 3: Collect pseudo-labels from trained RL agent
    print("\nCollecting pseudo-labels...")
    policy.eval()
    env.reset()
    selected_images = []
    selected_labels = []
    
    with torch.no_grad():
        for i in range(len(small_subset)):
            obs = env._get_obs()
            if np.all(obs == 0):  # End of episode
                break
                
            obs_v = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            logp = policy(obs_v)
            probs = logp.exp().cpu().numpy().squeeze(0)
            action = np.argmax(probs)  # Greedy action
            
            if action == 1:  # Accept pseudo-label
                x, y_true = small_subset[i]
                model.eval()
                with torch.no_grad():
                    logits = model(x.unsqueeze(0).to(device))
                    probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
                pred = probs.argmax()
                confidence = probs.max()
                
                if confidence >= 0.3:
                    selected_images.append(x)
                    selected_labels.append(pred)
            
            env.ptr += 1
            if env.ptr >= len(small_subset):
                break

    print(f"Selected {len(selected_images)} pseudo-labels out of {len(small_subset)}")

    # STEP 4: Train on combined dataset
    if len(selected_images) > 0:
        print("Training on combined dataset...")
        # Combine labeled and pseudo-labeled data
        labeled_x = torch.stack([x for x, y in labeled_subset])
        labeled_y = torch.tensor([y for x, y in labeled_subset])
        
        pseudo_x = torch.stack(selected_images)
        pseudo_y = torch.tensor(selected_labels)
        
        combined_x = torch.cat([labeled_x, pseudo_x])
        combined_y = torch.cat([labeled_y, pseudo_y])
        
        combined_dataset = TensorDataset(combined_x, combined_y)
        combined_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)
        
        # Train on combined dataset
        model.train()
        for epoch in range(10):
            for x, y in combined_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
            print(f"Combined training epoch {epoch+1}/10, loss: {loss.item():.4f}")

    # STEP 5: Final evaluation
    print("\nFinal evaluation...")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    final_acc = correct / total
    print(f"Final model accuracy: {final_acc:.4f}")
    print(f"Improvement from pseudo-labeling: {final_acc - initial_acc:.4f}")

    # log final accuracy
    with open("results_step4_rl.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.datetime.now().isoformat(), "final_accuracy", final_acc])
