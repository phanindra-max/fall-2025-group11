# -*- coding: utf-8 -*-
"""
Author: Prudhvi Chekuri
Date: 2025-09-15
Version: 1.0
"""


import os
import random
import tensorflow as tf
import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout



def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across numpy, tensorflow, and python's random module.
    :param seed: int - The seed value to set (default=42)
    :rtype: None
    """
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # os.environ["TF_DETERMINISTIC_OPS"] = "1" # add if needed for more determinism



def visualize_training_data(
    x_data: np.ndarray, 
    y_data: np.ndarray, 
    n: int = 25
) -> Figure:
    """
    Display n random images from the dataset with their labels.

    :param x_data: np.ndarray - Image data (numpy array, e.g., MNIST images)
    :param y_data: np.ndarray - Corresponding labels
    :param n: int - Number of images to display (default=25)
    :rtype: Figure
    :return: Matplotlib Figure object containing the displayed images
    """
    indices = np.random.choice(len(x_data), n, replace=False)
    rows = cols = int(np.sqrt(n))

    fig = plt.figure(figsize=(10, 10))
    for i, idx in enumerate(indices):
        plt.subplot(rows, cols, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_data[idx], cmap=plt.cm.binary)
        plt.xlabel(str(y_data[idx]))
    plt.show()

    return fig



def load_and_preprocess_mnist() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load the MNIST dataset, normalize pixel values, 
    and add a channel dimension.

    :rtype: Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    :return: ((x_train, y_train), (x_test, y_test)) with preprocessed data
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension for CNNs (28, 28, 1)
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    return (x_train, y_train), (x_test, y_test)



def build_and_compile_cnn_model(
    input_shape: tuple[int, int, int] = (28, 28, 1),
    conv_filters: list[int] = [32, 64],
    kernel_size: tuple[int, int] = (3, 3),
    pool_size: tuple[int, int] = (2, 2),
    dense_units: int = 128,
    dropout_rate: float = 0.5,
    num_classes: int = 10,
    final_activation: str = 'softmax',
    compile: bool = True
) -> Sequential:
    """
    Dynamically (With conv block (Conv2D and MaxPooling2D) stack followed by Flatten and Dense) build a CNN model.

    :param input_shape: tuple[int, int, int] - Shape of input images (default: MNIST (28, 28, 1))
    :param conv_filters: list[int] - List of filters for each Conv2D block
    :param kernel_size: tuple[int, int] - Kernel size for Conv2D layers
    :param pool_size: tuple[int, int] - Pool size for MaxPooling2D layers
    :param dense_units: int - Number of units in the dense layer
    :param dropout_rate: float - Dropout rate for regularization
    :param num_classes: int - Number of output classes
    :rtype: Sequential
    :return: A compiled Keras Sequential CNN model
    """
    model = Sequential()

    # First Conv block (with input shape)
    model.add(Conv2D(filters=conv_filters[0], kernel_size=kernel_size,
                     activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=pool_size))

    # Remaining Conv blocks
    for f in conv_filters[1:]:
        model.add(Conv2D(filters=f, kernel_size=kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=pool_size))

    # Flatten → Dense → Dropout → Output
    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation=final_activation))

    # Compile model
    if compile:
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model



def train_and_evaluate(
    model: Sequential,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    epochs: int = 10,
    validation_split: float = 0.1,
    verbose: int = 1
) -> Tuple[History, float, float]:
    """
    Train and evaluate a CNN model.

    :param model: Sequential - Keras Sequential model
    :param x_train: np.ndarray - Training images
    :param y_train: np.ndarray - Training labels
    :param x_test: np.ndarray - Test images
    :param y_test: np.ndarray - Test labels
    :param batch_size: int - Batch size for training
    :param epochs: int - Number of epochs
    :param validation_split: float - Fraction of training data for validation
    :param verbose: int - Verbosity level for training
    :rtype: Tuple[History, float, float]
    :return: (training history, test_loss, test_accuracy)
    """
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        verbose=verbose
    )

    print("\n--- Evaluating Model on Test Data ---")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=verbose)
    print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    return history, test_loss, test_accuracy



def plot_training_history(history: History, title_prefix: str) -> Figure:
    """
    Plot training & validation accuracy and loss from a Keras History object.

    :param history: History - Training history returned by model.fit()
    :param title_prefix: str - Optional prefix for the plot titles
    :rtype: Figure
    :return: Matplotlib Figure object containing the plots
    """
    fig = plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{title_prefix} Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()
    return fig



def create_labeled_unlabeled_split(
    x_data: np.ndarray,
    y_data: np.ndarray,
    labeled_size: float = 0.1,
    random_state: int = 42
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Splits the training data into a small labeled set and a large unlabeled set.

    :param x_data: np.ndarray - Training features
    :param y_data: np.ndarray - Training labels
    :param labeled_size: float - Proportion or absolute number of samples for the labeled set
    :param random_state: int - Seed for reproducibility
    :rtype: Tuple[Tuple, Tuple]
    :return: A tuple containing ((x_labeled, y_labeled), (x_unlabeled, y_unlabeled))
    """
    x_labeled, x_unlabeled, y_labeled, y_unlabeled = train_test_split(
        x_data, y_data, train_size=labeled_size, stratify=y_data, random_state=random_state
    )
    print(f"Labeled data shape: {x_labeled.shape}")
    print(f"Unlabeled data shape: {x_unlabeled.shape}")
    return (x_labeled, y_labeled), (x_unlabeled, y_unlabeled)



def generate_pseudo_labels(
    model: Sequential, 
    x_unlabeled: np.ndarray, 
    confidence_threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates pseudo-labels for unlabeled data based on a confidence threshold.

    :param model: Sequential - The trained model to make predictions
    :param x_unlabeled: np.ndarray - The data to be pseudo-labeled
    :param confidence_threshold: float - The probability threshold to accept a prediction
    :rtype: Tuple[np.ndarray, np.ndarray]
    :return: (x_pseudo, y_pseudo) - Data and pseudo-labels for confident predictions
    """
    print("\n--- Generating Pseudo-Labels ---")
    predictions = model.predict(x_unlabeled)
    
    pseudo_labels = np.argmax(predictions, axis=1)
    max_probabilities = np.max(predictions, axis=1)
    
    confident_mask = max_probabilities > confidence_threshold
    
    x_pseudo = x_unlabeled[confident_mask]
    y_pseudo = pseudo_labels[confident_mask]
    
    print(f"Generated {len(x_pseudo)} pseudo-labels with confidence > {confidence_threshold}")
    return x_pseudo, y_pseudo



def plot_comparison_history(
    histories: Dict[str, History], 
    title_prefix: str = ""
) -> Figure:
    """
    Plots a comparison of training/validation accuracy and loss from multiple History objects.

    :param histories: Dict[str, History] - A dictionary mapping a label to a History object
    :param title_prefix: str - Optional prefix for the plot titles
    :rtype: Figure
    :return: Matplotlib Figure object containing the plots
    """
    fig = plt.figure(figsize=(14, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(history.history["val_accuracy"], label=f"{name} Validation", linestyle='--')
        plt.plot(history.history["accuracy"], label=f"{name} Training")
    plt.title(f"{title_prefix} Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot loss
    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(history.history["val_loss"], label=f"{name} Validation", linestyle='--')
        plt.plot(history.history["loss"], label=f"{name} Training")
    plt.title(f"{title_prefix} Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
    return fig



def plot_iterative_history(histories: Dict[str, History], title_prefix: str = "") -> None:
    """
    Plots training and validation accuracy and loss across multiple iterations.
    
    :param histories: Dict[str, History] - A dictionary mapping iteration names to History objects
    :param title_prefix: str - Optional prefix for the plot titles
    :rtype: None
    """
    plt.figure(figsize=(14, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(history.history['val_accuracy'], label=f'{name} Val', linestyle='--')
    plt.title(f'{title_prefix} Validation Accuracy Across Iterations')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.5)

    # Loss Plot
    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(history.history['val_loss'], label=f'{name} Val', linestyle='--')
    plt.title(f'{title_prefix} Validation Loss Across Iterations')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.show()



def plot_accuracy_improvement(accuracies: List[float], title_prefix: str = "") -> None:
    
    """
    Plots the improvement in test accuracy over multiple iterations.
    :param accuracies: List[float] - List of test accuracies from each iteration
    :param title_prefix: str - Optional prefix for the plot title
    :rtype: None
    """
    
    plt.figure(figsize=(8, 5))
    iterations = range(len(accuracies))
    plt.plot(iterations, accuracies, marker='o', linestyle='-')
    plt.title(f'{title_prefix} Test Accuracy Improvement Over Iterations')
    plt.xlabel('Iteration (0 = Initial Model)')
    plt.ylabel('Test Accuracy')
    plt.xticks(iterations)
    plt.grid(True)
    plt.show()
    
    
    
@tf.function
def weak_augment(image):
    """Applies a simple augmentation: random translation.
    :param image: tf.Tensor - Input image tensor
    :rtype: tf.Tensor
    :return: Augmented image tensor
    """
    dx = tf.random.uniform(shape=[], minval=-2, maxval=2, dtype=tf.int32)
    dy = tf.random.uniform(shape=[], minval=-2, maxval=2, dtype=tf.int32)
    return tf.roll(image, shift=[dy, dx], axis=[0, 1])



@tf.function
def strong_augment(image):
    """Applies a stronger augmentation: translation and flip.
    :param image: tf.Tensor - Input image tensor
    :rtype: tf.Tensor
    :return: Augmented image tensor
    """
    image = weak_augment(image) # Start with weak augmentation
    image = tf.image.random_flip_left_right(image)
    # You could add more here, like random rotation or cutout, for even stronger augmentation
    return image



def smooth_curve(points, factor=0.8):
    """
    Smooths a list of points using exponential moving average.
    :param points: List[float] - List of points to smooth
    :param factor: float - Smoothing factor (0 < factor < 1)
    :rtype: List[float]
    :return: Smoothed list of points
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points