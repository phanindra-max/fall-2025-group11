# -*- coding: utf-8 -*-
"""
Author: Prudhvi Chekuri
Date: 2025-09-15
Version: 1.0
"""



import numpy as np
from mnist_training_utils import (load_and_preprocess_mnist, build_and_compile_cnn_model, train_and_evaluate, 
                                  plot_comparison_history, create_labeled_unlabeled_split, generate_pseudo_labels,
                                  set_random_seed)



def main(labeled_size: int = 1000, epochs: int = 10, batch_size: int = 32, validation_split: float = 0.1, 
         confidence_threshold: float = 0.95, verbose: int = 1):
    """
    Main function to perform pseudo-labeling on MNIST dataset.
    
    :param labeled_size: int - Number of labeled samples to use
    :param epochs: int - Number of epochs for training
    :param batch_size: int - Batch size for training
    :param validation_split: float - Fraction of training data to use for validation
    :param confidence_threshold: float - Confidence threshold for pseudo-labeling
    :param verbose: int - Verbosity level for training
    :rtype: None
    """
    print("\n" + "="*50)
    print("🚀 STARTING: Pseudo-Labeling Experiment")
    print("="*50)

    # 0. Set random seed for reproducibility
    set_random_seed(42)

    # 1. Load and split data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()
    (x_labeled, y_labeled), (x_unlabeled, _) = create_labeled_unlabeled_split(x_train, y_train, labeled_size=labeled_size)
    print(f"Split data into {len(x_labeled)} labeled and {len(x_unlabeled)} unlabeled samples.")
    
    # 2. Train initial model on the small labeled set
    print("\n--- Step 1: Training initial model on labeled data ---")
    initial_model = build_and_compile_cnn_model()
    history_initial, _, _ = train_and_evaluate(initial_model, x_labeled, y_labeled, x_test, y_test, epochs=epochs, 
                                               batch_size=batch_size, validation_split=validation_split, verbose=verbose)
    
    # 3. Generate pseudo-labels
    print("\n--- Step 2: Generating pseudo-labels from unlabeled data ---")
    x_pseudo, y_pseudo = generate_pseudo_labels(initial_model, x_unlabeled, confidence_threshold)
    print(f"Generated {len(y_pseudo)} new pseudo-labels.")

    # 4. Combine datasets and train final model
    print("\n--- Step 3: Training final model on combined data ---")
    x_combined = np.concatenate([x_labeled, x_pseudo], axis=0)
    y_combined = np.concatenate([y_labeled, y_pseudo], axis=0)
    final_model = build_and_compile_cnn_model() # Re-initialize model
    history_final, _, _ = train_and_evaluate(final_model, x_combined, y_combined, x_test, y_test, epochs=epochs, 
                                             batch_size=batch_size, validation_split=validation_split, verbose=verbose)
    
    # 5. Plot comparison
    plot_comparison_history({"Initial (Labeled Only)": history_initial,"Final (Pseudo-Labeled)": history_final}, title_prefix="Pseudo-Labeling MNIST")
    print("\n✅ COMPLETED: Pseudo-Labeling Experiment.\n")


if __name__ == "__main__":
    labeled_size = 1000
    epochs = 10
    batch_size = 128
    validation_split = 0.1
    confidence_threshold = 0.95
    verbose = 1
    main(labeled_size=labeled_size, epochs=epochs, batch_size=batch_size, validation_split=validation_split, 
         confidence_threshold=confidence_threshold, verbose=verbose)