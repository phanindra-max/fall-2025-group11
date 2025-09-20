# Deepika Baseline

## **Overview**

This folder contains the **baseline implementation** for our RL-based pseudo-labeling project. The goal is to generate **soft pseudo-labels** for unlabeled data using a **warm-up downstream model**, which serves as the foundation for later reinforcement learning (RL) improvements.

This baseline demonstrates how unlabeled data can be leveraged in a **semi-supervised learning setting**.

---

## **Key Features**

1. **Warm-up Training**

   * Train a small CNN/MLP model on a **labeled subset** of the dataset.
   * Provides initial knowledge for generating pseudo-labels.

2. **Soft Pseudo-Label Generation**

   * Predicts **probabilities** for each unlabeled sample.
   * Maintains uncertainty information (soft labels).

3. **Ready for Downstream Experiments**

   * Soft pseudo-labels can be used to:

     * Train downstream models
     * Evaluate RL-based selection strategies
     * Compare with baseline accuracy

---

## **Dependencies**

Install required packages:

```bash
pip install torch torchvision numpy matplotlib
```

---

## **Next Steps**

* Integrate **RL agent** to select which pseudo-labels to accept
* Apply **mixup augmentation** during pseudo-label training
* Add **RA/RH regularizers** to prevent label collapse
* Combine accepted pseudo-labels with labeled data for downstream model training
* Evaluate improvements over baseline

---

## **References**

* Arazo et al., “Pseudo-Labeling with Class Balance and Uncertainty Regularization,” 2019. [arXiv:1908.02983](https://arxiv.org/pdf/1908.02983)
* OpenAI Gym / Gymnasium documentation (for RL environment inspiration)

---


