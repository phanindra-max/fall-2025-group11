# -*- coding: utf-8 -*-
"""
Author: Prudhvi Chekuri
Date: 2025-09-15
Version: 1.0
"""



import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ssl_models import FixMatch, UDAModel, PseudoLabeling
from mnist_training_utils import (load_and_preprocess_mnist, build_and_compile_cnn_model, set_random_seed, 
                                  smooth_curve)



def main(seed=42, labeled_size=1000, test_size=0.1, val_batch_size=64, verbose=1, fixmatch_batch_size=64, learning_rate=1e-3,
         fixmatch_steps_per_epoch=105, uda_steps_per_epoch=105, fixmatch_epochs=30, uda_epochs=30):
    """
    Runs a fair comparison (As much as possible) between Pseudo-Labeling, FixMatch, and UDA.
    
    :param seed: int - Random seed for reproducibility
    :param labeled_size: int - Number of labeled samples to use from the training set
    :param test_size: float - Proportion of labeled data to use as validation set
    :param val_batch_size: int - Batch size for validation dataset
    :param verbose: int - Verbosity level for training outputs
    :param fixmatch_batch_size: int - Total batch size for FixMatch training (split between labeled and unlabeled)
    :param learning_rate: float - Learning rate for optimizers
    :param fixmatch_steps_per_epoch: int - Steps per epoch for FixMatch training
    :param uda_steps_per_epoch: int - Steps per epoch for UDA training
    :param fixmatch_epochs: int - Number of epochs to train FixMatch
    :param uda_epochs: int - Number of epochs to train UDA
    :rtype: None
    """
    
    # =============================================================================
    # 1. UNIFIED SETUP & DATA PREPARATION
    # =============================================================================

    # Set random seeds for reproducibility
    set_random_seed(seed)

    # Load and preprocess MNIST data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()

    # Reduce labeled data to labeled_size samples
    x_labeled, x_unlabeled, y_labeled, _ = train_test_split(x_train, y_train, train_size=labeled_size, stratify=y_train, 
                                                            random_state=seed)

    # Create a smaller validation set from the labeled data
    x_train_lab, x_val_lab, y_train_lab, y_val_lab = train_test_split(x_labeled, y_labeled, test_size=test_size, 
                                                                      stratify=y_labeled, random_state=seed)

    print(f"Total Labeled: {x_labeled.shape[0]}, Total Unlabeled: {x_unlabeled.shape[0]}")
    print(f"Training Labeled: {x_train_lab.shape[0]}, Validation Labeled: {x_val_lab.shape[0]}")

    # Create a validation dataset for use in all models
    val_ds = tf.data.Dataset.from_tensor_slices((x_val_lab, y_val_lab)).batch(val_batch_size)

    # Dictionary to store results from all experiments
    results = {}


    # ===================================================================
    # 2. ALGORITHM 1: PSEUDO-LABELING
    # ===================================================================
    print("\n--- Running Pseudo-Labeling ---")
    pl_trainer = PseudoLabeling()

    # The fit method now handles all the complex logic internally.
    final_model_pl, history_pl_initial, history_pl_final = pl_trainer.fit(
                                                            x_train_lab, y_train_lab, x_unlabeled, validation_data=val_ds)

    # Evaluate and store results
    loss_initial, acc_initial = pl_trainer.initial_model.evaluate(x_test, y_test, verbose=verbose)
    loss_final, acc_final = final_model_pl.evaluate(x_test, y_test, verbose=verbose)
    print(f"Pseudo-Labeling Initial Accuracy: {acc_initial*100:.2f}%")
    print(f"Pseudo-Labeling Final Accuracy: {acc_final*100:.2f}%")

    results['Pseudo-Labeling'] = {
       'initial_acc': acc_initial, 'final_acc': acc_final,
       'history_initial': history_pl_initial.history, 'history_final': history_pl_final.history
    }


    # ===================================================================
    # 3. ALGORITHM 2: FIXMATCH
    # ===================================================================
    print("\n--- Running FixMatch ---")
    
    labeled_batch_size = fixmatch_batch_size // 4
    unlabeled_batch_size = fixmatch_batch_size - labeled_batch_size

    labeled_ds_fm = tf.data.Dataset.from_tensor_slices((x_train_lab, y_train_lab)).shuffle(len(x_train_lab)).repeat().batch(labeled_batch_size)
    unlabeled_ds_fm = tf.data.Dataset.from_tensor_slices(x_unlabeled).shuffle(len(x_unlabeled)).repeat().batch(unlabeled_batch_size)
    train_ds_fm = tf.data.Dataset.zip((labeled_ds_fm, unlabeled_ds_fm))

    
    base_model_fm = build_and_compile_cnn_model(dense_units=64, final_activation=None, compile=False)
    fixmatch_model = FixMatch(base_model_fm)
    fixmatch_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    
    print("Training FixMatch model...")
    history_fm = fixmatch_model.fit(train_ds_fm, steps_per_epoch=fixmatch_steps_per_epoch, epochs=fixmatch_epochs, validation_data=val_ds, verbose=verbose)
    print("FixMatch training complete.")

    # Evaluate final model
    fm_results = fixmatch_model.evaluate(x_test, y_test, return_dict=True, verbose=verbose)
    print(f"Final FixMatch Test Accuracy: {fm_results['accuracy']*100:.2f}%")
    results['FixMatch'] = {'final_acc': fm_results['accuracy'], 'history': history_fm.history}


    # ===================================================================
    # 4. ALGORITHM 3: UDA
    # ===================================================================
    print("\n--- Running UDA ---")
    
    # The large class definition is gone!
    base_model_uda = build_and_compile_cnn_model(dense_units=64, final_activation=None, compile=False)
    uda_model = UDAModel(base_model_uda) # Clean and simple
    uda_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

    print("Training UDA model...")
    history_uda = uda_model.fit(train_ds_fm, steps_per_epoch=uda_steps_per_epoch, epochs=uda_epochs, validation_data=val_ds, verbose=verbose)
    print("UDA training complete.")

    # Evaluate final model
    uda_results = uda_model.evaluate(x_test, y_test, return_dict=True, verbose=verbose)
    print(f"Final UDA Test Accuracy: {uda_results['accuracy']*100:.2f}%")

    results['UDA'] = {'final_acc': uda_results['accuracy'], 'history': history_uda.history}


    # ===================================================================
    # 5. FINAL RESULTS & VISUALIZATION
    # ===================================================================

    print("\n--- Final Test Accuracy Summary ---")
    for name, res in results.items():
        if 'initial_acc' in res:
            print(f"{name}: Initial={res['initial_acc']*100:.2f}%, Final={res['final_acc']*100:.2f}%")
        else:
            print(f"{name}: Final={res['final_acc']*100:.2f}%")
            

    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # --- Accuracy Plot (by Epoch) ---
    ax1.set_title('Validation Accuracy vs. Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (Smoothed)')
    ax1.plot(smooth_curve(results['Pseudo-Labeling']['history_initial']['val_accuracy']), label='PL (Initial)', linestyle=':')
    ax1.plot(smooth_curve(results['Pseudo-Labeling']['history_final']['val_accuracy']), label='PL (Final)', marker='o', markersize=3)
    ax1.plot(smooth_curve(results['FixMatch']['history']['val_accuracy']), label='FixMatch', marker='s', markersize=3)
    ax1.plot(smooth_curve(results['UDA']['history']['val_accuracy']), label='UDA', marker='^', markersize=3)
    ax1.set_title('Validation Accuracy vs. Epoch for SSL Methods Comparison')
    ax1.legend()

    # --- Loss Plot (by Epoch) ---
    ax2.set_title('Validation Loss vs. Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (Smoothed)')
    ax2.plot(smooth_curve(results['Pseudo-Labeling']['history_initial']['val_loss']), label='PL (Initial)', linestyle=':')
    ax2.plot(smooth_curve(results['Pseudo-Labeling']['history_final']['val_loss']), label='PL (Final)', marker='o', markersize=3)
    ax2.plot(smooth_curve(results['FixMatch']['history']['val_loss']), label='FixMatch', marker='s', markersize=3)
    ax2.plot(smooth_curve(results['UDA']['history']['val_loss']), label='UDA', marker='^', markersize=3)
    ax2.set_title('Validation Loss vs. Epoch for SSL Methods Comparison')
    ax2.legend()

    plt.tight_layout()
    plt.show()
    
    
    
if __name__ == '__main__':
    seed = 42
    labeled_size = 1000
    test_size = 0.1
    val_batch_size = 64
    verbose = 1
    fixmatch_batch_size = 64
    learning_rate = 1e-3
    fixmatch_steps_per_epoch = 105
    uda_steps_per_epoch = 105
    fixmatch_epochs = 30
    uda_epochs = 30
    
    main(seed, labeled_size, test_size, val_batch_size, verbose, fixmatch_batch_size, learning_rate,
         fixmatch_steps_per_epoch, uda_steps_per_epoch, fixmatch_epochs, uda_epochs)