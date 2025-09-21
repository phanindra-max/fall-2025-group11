# paste this entire file (or replace existing run_experiment) and run:
# python pseudo_labeling_step3_iterative.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import random
import time
import csv
import os
from datetime import datetime

# -------------------------
# Config / hyperparameters
# -------------------------
seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
num_classes = 10
warmup_epochs = 1
downstream_epochs = 2
mixup_alpha = 0.4
confidence_threshold = 0.9
lambda_reg = 0.1
num_iterations = 3
num_labeled = 1000
num_unlabeled = 1000
results_file = "results.csv"

random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if device.startswith("cuda"):
    torch.cuda.manual_seed_all(seed)

# -------------------------
# Simple model (small CNN)
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3, 1)
        self.fc = nn.Linear(26*26*32, num_classes)
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# -------------------------
# Helpers
# -------------------------
def mixup_data(x, y, alpha=mixup_alpha):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def ra_rh_regularizer(preds):
    mean_pred = preds.mean(dim=0)
    ra = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8))
    sample_entropy = -torch.sum(preds * torch.log(preds + 1e-8), dim=1)
    rh = -sample_entropy.mean()
    return ra + rh

# -------------------------
# Data
# -------------------------
transform = transforms.Compose([transforms.ToTensor()])
full_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

labeled_indices = list(range(num_labeled))
labeled_subset = Subset(full_train, labeled_indices)
labeled_loader = DataLoader(labeled_subset, batch_size=batch_size, shuffle=True)

unlabeled_indices = list(range(num_labeled, num_labeled + num_unlabeled))
unlabeled_subset = [full_train[i] for i in unlabeled_indices]   # list of (x, y)

# -------------------------
# Utility functions
# -------------------------
def train_warmup(model, opt, epochs=1):
    model.train()
    for ep in range(epochs):
        for x, y in labeled_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

def generate_soft_labels(model, pool):
    model.eval()
    probs_list = []
    with torch.no_grad():
        for x, y in pool:
            t = x.unsqueeze(0).to(device)
            logits = model(t)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu()
            probs_list.append(probs)
    return torch.stack(probs_list)

def evaluate(model):
    model.eval()
    correct = 0; total = 0
    loader = DataLoader(test_set, batch_size=batch_size)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# -------------------------
# Main experiment
# -------------------------
def run_experiment():
    start_time = time.time()
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    backbone = SimpleCNN(num_classes).to(device)
    opt = torch.optim.Adam(backbone.parameters(), lr=1e-3)

    # Warm-up
    train_warmup(backbone, opt, epochs=warmup_epochs)

    remaining_pool = unlabeled_subset.copy()
    total_accepted = 0
    iter_records = []   # <--- ensure this is defined BEFORE the loop

    for it in range(1, num_iterations+1):
        if len(remaining_pool) == 0:
            print("No remaining unlabeled samples. Stopping.")
            break
        print(f"\nIteration {it} — remaining pool size: {len(remaining_pool)}")

        soft = generate_soft_labels(backbone, remaining_pool)  # (M, C)
        max_vals, _ = torch.max(soft, dim=1)
        selected_mask = max_vals >= confidence_threshold
        num_selected = int(selected_mask.sum().item())
        print(f"Selected {num_selected}/{len(remaining_pool)} by confidence >= {confidence_threshold}")

        if num_selected == 0:
            print("No high-confidence samples selected; stopping early.")
            break

        selected_images = torch.stack([remaining_pool[i][0] for i in range(len(remaining_pool)) if selected_mask[i]])
        selected_labels_soft = torch.stack([soft[i] for i in range(len(soft)) if selected_mask[i]])
        selected_labels_hard = torch.argmax(selected_labels_soft, dim=1)

        sel_true = torch.tensor([remaining_pool[i][1] for i in range(len(remaining_pool)) if selected_mask[i]])
        pseudo_error = (selected_labels_hard.cpu() != sel_true).float().mean().item() if num_selected>0 else None

        labeled_x = torch.stack([full_train[i][0] for i in labeled_indices])
        labeled_y = torch.tensor([full_train[i][1] for i in labeled_indices])
        combined_x = torch.cat([labeled_x, selected_images.cpu()])
        combined_y = torch.cat([labeled_y, selected_labels_hard.cpu()])

        train_dataset = TensorDataset(combined_x, combined_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        backbone.train()
        opt = torch.optim.Adam(backbone.parameters(), lr=1e-3)
        for epoch in range(downstream_epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                xb, ya, yb_mix, lam = mixup_data(xb, yb, alpha=mixup_alpha)
                opt.zero_grad()
                logits = backbone(xb)
                loss = lam * F.cross_entropy(logits, ya) + (1-lam) * F.cross_entropy(logits, yb_mix)
                probs = F.softmax(logits, dim=1)
                loss += lambda_reg * ra_rh_regularizer(probs)
                loss.backward()
                opt.step()

        new_pool = [remaining_pool[i] for i in range(len(remaining_pool)) if not selected_mask[i]]
        remaining_pool = new_pool
        total_accepted += num_selected

        test_acc = evaluate(backbone)
        print(f"After iteration {it}, test accuracy: {test_acc:.4f}, pseudo_error: {pseudo_error}")

        iter_records.append({
            "iteration": it,
            "num_selected": num_selected,
            "pseudo_error": pseudo_error if pseudo_error is not None else "",
            "remaining": len(remaining_pool),
            "test_acc": test_acc
        })

    final_acc = evaluate(backbone)
    elapsed = (time.time() - start_time)/60.0

    # Summary row (single-line)
    result_row = {
        "experiment": "pseudo_labeling_step3_iterative",
        "seed": seed,
        "dataset": "MNIST",
        "methods": "confidence+mixup+RA_RH+iterative",
        "tau": confidence_threshold,
        "mixup_alpha": mixup_alpha,
        "lambda_ra_rh": lambda_reg,
        "iterations_ran": len(iter_records),
        "num_labeled": num_labeled,
        "num_pseudo_labels": total_accepted,
        "test_accuracy": final_acc,
        "training_time_minutes": round(elapsed, 2),
        "notes": ""
    }

    # Append summary to results.csv (create header if missing)
    results_path = os.path.join(save_dir, results_file)
    file_exists = os.path.exists(results_path)
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_row)

    # Print per-iteration results
    print("\nIteration-wise results:")
    print(f"{'Iter':<5}{'#Sel':<8}{'PseudoErr':<12}{'Remain':<10}{'TestAcc':<10}")
    for rec in iter_records:
        pe = rec['pseudo_error'] if rec['pseudo_error'] != "" else 0.0
        print(f"{rec['iteration']:<5}{rec['num_selected']:<8}{pe:<12.4f}{rec['remaining']:<10}{rec['test_acc']:<10.4f}")

    # Save per-iteration CSV with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    iter_results_path = os.path.join(save_dir, f"iteration_results_step3_{timestamp}.csv")
    with open(iter_results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "num_selected", "pseudo_error", "remaining", "test_acc"])
        writer.writeheader()
        writer.writerows(iter_records)

    print(f"\nIteration results saved to {iter_results_path}")

    # Final summary print
    print("\nFinal summary:")
    print("Experiment:", result_row["experiment"])
    print("Dataset:", result_row["dataset"])
    print("Total pseudo-labels accepted:", total_accepted)
    print("Final test accuracy:", f"{final_acc:.4f}")
    print("Training time (min):", round(elapsed, 2))

    return iter_records, result_row

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    records, summary = run_experiment()
