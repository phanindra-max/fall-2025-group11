# -*- coding: utf-8 -*-
"""
Author: Prudhvi Chekuri
Date: 2025-09-15
Version: 1.0
"""



from mnist_training_utils import (load_and_preprocess_mnist, build_and_compile_cnn_model, train_and_evaluate, 
                                  plot_training_history, set_random_seed)



def main(epochs: int = 10, batch_size: int = 32, validation_split: float = 0.1, verbose: int = 1):
    """
    Main function to train a standard CNN on the full MNIST dataset.
    
    :param epochs: int - Number of epochs for training
    :param batch_size: int - Batch size for training
    :param validation_split: float - Fraction of training data to use for validation
    :param verbose: int - Verbosity level for training
    :rtype: None
    """
    print("\n" + "="*50)
    print("🚀 STARTING: Standard CNN Training on Full Dataset")
    print("="*50)
    
    # 0. Set random seed for reproducibility
    set_random_seed(42)
    
    # 1. Load data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()

    # 2. Build and train model
    model = build_and_compile_cnn_model()
    model.summary()
    history, _, _ = train_and_evaluate(model, x_train, y_train, x_test, y_test, epochs=epochs, 
                                       batch_size=batch_size, validation_split=validation_split, 
                                       verbose=verbose)

    # 3. Plot results
    plot_training_history(history, title_prefix="Standard CNN MNIST")
    print("\n✅ COMPLETED: Standard CNN Training.\n")
    
if __name__ == "__main__":
    epochs = 10
    batch_size = 128
    validation_split = 0.1
    verbose = 1
    main(epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose)