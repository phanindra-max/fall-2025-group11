# -*- coding: utf-8 -*-
"""
Author: Prudhvi Chekuri
Date: 2025-09-15
Version: 1.0
"""



from mnist_training_utils import (load_and_preprocess_mnist, build_and_compile_cnn_model, plot_iterative_history, 
                                  plot_accuracy_improvement, set_random_seed, create_labeled_unlabeled_split)
import numpy as np



def main(
    labeled_size: int = 1000,
    initial_batch_size: int = 128,
    initial_epochs: int = 10,
    initial_validation_split: float = 0.1,
    final_batch_size: int = 128,
    final_epochs: int = 10,
    final_validation_split: float = 0.1,
    n_iterations: int = 3,
    confidence_threshold: float = 0.99,
    seed: int = 42,
    verbose: int = 1
):
    """
    Main function to perform iterative pseudo-labeling on MNIST dataset.
    :param labeled_size: int - Initial number of labeled samples
    :param initial_batch_size: int - Batch size for initial training
    :param initial_epochs: int - Number of epochs for initial training
    :param initial_validation_split: float - Validation split for initial training
    :param final_batch_size: int - Batch size for iterative training
    :param final_epochs: int - Number of epochs for iterative training
    :param final_validation_split: float - Validation split for iterative training
    :param n_iterations: int - Number of pseudo-labeling iterations
    :param confidence_threshold: float - Confidence threshold for pseudo-labeling
    :param seed: int - Random seed for reproducibility
    :param verbose: int - Verbosity level for training
    :rtype: None
    """
    print("\n" + "="*50)
    print("🚀 STARTING: Iterative Pseudo-Labeling Experiment")
    print("="*50)
    
    # 0. Set random seed for reproducibility
    set_random_seed(seed)

    # 1. Load and split data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()
    (x_labeled, y_labeled), (x_unlabeled, _) = create_labeled_unlabeled_split(x_train, y_train, labeled_size=labeled_size, random_state=seed)
    print(f"Initial split: {len(x_labeled)} labeled, {len(x_unlabeled)} unlabeled.")
    
    # 2. Train initial model
    print("\n--- Training Initial Model on Labeled Data ---")
    model = build_and_compile_cnn_model()
    history_initial = model.fit(x_labeled, y_labeled, batch_size=initial_batch_size, epochs=initial_epochs, validation_split=initial_validation_split, verbose=verbose)
    _, initial_acc = model.evaluate(x_test, y_test, verbose=verbose)
    print(f"Initial model test accuracy: {initial_acc*100:.2f}%")

    histories = {"Initial": history_initial}
    test_accuracies = [initial_acc]
    
    x_unlabeled_current = x_unlabeled.copy()
    x_labeled_current, y_labeled_current = x_labeled.copy(), y_labeled.copy()

    # 3. Start Iterative Loop
    for i in range(n_iterations):
        print(f"\n--- PSEUDO-LABELING ITERATION {i + 1}/{n_iterations} ---")
        
        # Generate pseudo-labels from the remaining unlabeled pool
        predictions = model.predict(x_unlabeled_current, batch_size=256)
        pseudo_labels = np.argmax(predictions, axis=1)
        max_probs = np.max(predictions, axis=1)
        
        confident_mask = max_probs > confidence_threshold
        x_pseudo = x_unlabeled_current[confident_mask]
        y_pseudo = pseudo_labels[confident_mask]

        if len(x_pseudo) == 0:
            print("No new pseudo-labels met the confidence threshold. Halting.")
            break
        
        print(f"Found {len(x_pseudo)} new pseudo-labels.")
        
        # Add new pseudo-labels to the labeled set for the next training round
        x_labeled_current = np.concatenate([x_labeled_current, x_pseudo])
        y_labeled_current = np.concatenate([y_labeled_current, y_pseudo])

        # Remove the used samples from the unlabeled pool
        x_unlabeled_current = x_unlabeled_current[~confident_mask]
        print(f"{len(x_unlabeled_current)} samples remaining in unlabeled pool.")

        # Retrain a new model on the augmented dataset
        model = build_and_compile_cnn_model()
        history_iter = model.fit(x_labeled_current, y_labeled_current, batch_size=final_batch_size, epochs=final_epochs, validation_split=final_validation_split, verbose=verbose)
        _, iter_acc = model.evaluate(x_test, y_test, verbose=verbose)
        print(f"Iteration {i+1} test accuracy: {iter_acc*100:.2f}%")
        
        histories[f'Iter {i+1}'] = history_iter
        test_accuracies.append(iter_acc)
        
        if len(x_unlabeled_current) == 0:
            print("Unlabeled data pool is empty. Halting.")
            break

    # 4. Plot results
    print("\n--- Plotting Experiment Results ---")
    plot_iterative_history(histories)
    plot_accuracy_improvement(test_accuracies)
    print("\n✅ COMPLETED: Iterative Pseudo-Labeling Experiment.\n")
    
    


if __name__ == "__main__":
    labeled_size = 1000
    initial_batch_size = 128
    initial_epochs = 10
    initial_validation_split = 0.1
    final_batch_size = 128
    final_epochs = 10
    final_validation_split = 0.1
    n_iterations = 3
    confidence_threshold = 0.99
    seed = 42
    verbose = 1

    main(
        labeled_size=labeled_size,
        initial_batch_size=initial_batch_size,
        initial_epochs=initial_epochs,
        initial_validation_split=initial_validation_split,
        final_batch_size=final_batch_size,
        final_epochs=final_epochs,
        final_validation_split=final_validation_split,
        n_iterations=n_iterations,
        confidence_threshold=confidence_threshold,
        seed=seed,
        verbose=verbose
    )