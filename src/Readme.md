### RLGSSL Source Overview

RLGSSL (Reinforcement Learning–Guided Semi-Supervised Learning) implementation with a teacher–student framework, mixup-based reward, and RL-weighted loss. This directory contains training pipelines, datasets, models, and RL components.

### Structure

```
src/
├── demo.py                    # Quick demo/testing
├── run_experiments.py         # CLI runner
├── train_rlgssl.py           # Main trainer
├── readme.md                 # Documentation
├── data/
│   └── ssl_datasets.py       # Data handling
├── models/
│   └── networks.py           # CNN-13, PolicyNetwork
└── rlgssl/
    ├── mixup.py              # Mixup generation
    ├── reward_function.py    # RL reward computation
    ├── rl_loss.py           # RL loss functions
    └── teacher_student.py    # EMA teacher framework
```

### Files and key classes/methods

- `train_rlgssl.py`
  - **RLGSSLConfig**: Training hyperparameters and runtime options (dataset, epochs, EMA, AMP, etc.).
  - **RLGSSLTrainer**: Full training loop combining warmup and RLGSSL phases; handles data, model, optimizer, AMP, checkpoints.

- `run_experiments.py`
  - CLI entrypoint to run a single/quick/paper-style experiment. Builds `RLGSSLConfig`, runs `RLGSSLTrainer` and saves results.

- `data/ssl_datasets.py`
  - **SSLDatasetSplitter**: Creates balanced labeled/unlabeled splits using dataset labels without loading images.
  - **SSLDataset**: Lightweight wrapper returning tensors for labeled/unlabeled samples.
  - **create_ssl_dataloaders(...)**: Returns `train_labeled`, `train_unlabeled`, and `test` loaders with performance flags.

- `models/networks.py`
  - **CNN13**: 13-layer CNN used as student/teacher backbone for CIFAR/SVHN-sized images.
  - **PolicyNetwork**: Wrapper exposing `get_logits` and `get_probabilities` for RL components.

- `rlgssl/teacher_student.py`
  - **TeacherStudentFramework**: Manages student/EMA-teacher models, supervised loss (CE), consistency loss (KL), evaluation, EMA update.
  - **warmup_teacher_student(...)**: Pre-training loop optimizing supervised+consistency losses before RLGSSL.

- `rlgssl/mixup.py`
  - **MixupGenerator**: Creates mixup pairs between labeled and pseudo-labeled data; handles size mismatch and mu sampling.

- `rlgssl/reward_function.py`
  - **RLRewardFunction**: Reward = negative MSE between model predictions on mixup data and mixup labels (no grad by default).
  - **AdaptiveRewardFunction**: Extends reward with confidence/diversity bonuses; returns components for logging/analysis.

- `rlgssl/rl_loss.py`
  - **RLLossFunction**: Computes RL loss `-E[KL(e, y^u)] * Reward` with uniform baseline weighting.
  - **AdaptiveRLLossFunction**: Confidence-aware curriculum and entropy regularization on top of RL loss.
  - **CombinedLossFunction**: Final objective `L = L_rl + λ1 L_sup + λ2 L_cons`.

- `demo.py`
  - Minimal script wiring components for a quick qualitative run (example usage/reference).

### Notes

- Core flow: data loaders → `TeacherStudentFramework` warmup → mixup + reward → RL loss weighting → combined objective → EMA updates.
- Models expose both logits and probabilities to support CE/KL (supervised/consistency) and RL components.

