# -*- coding: utf-8 -*-
"""
Author: Prudhvi Chekuri
Date: 2025-09-15
Version: 1.0
"""



import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from mnist_training_utils import weak_augment, strong_augment, build_and_compile_cnn_model



class PseudoLabeling:
    """
    Implements the Pseudo-Labeling semi-supervised learning approach.
    """
    
    def __init__(self, confidence_threshold=0.95):
        """
        Args:
            confidence_threshold (float): The probability threshold for accepting pseudo-labels.
        """
        self.confidence_threshold = confidence_threshold
        self.initial_model = None
        self.final_model = None

    def fit(self, x_train_lab, y_train_lab, x_unlabeled, validation_data,
            initial_epochs=10, initial_steps=15, final_epochs=20, final_steps=150, verbose=1):
        """
        Trains the model using the Pseudo-Labeling approach.
        :param x_train_lab: Labeled training data.
        :param y_train_lab: Labels for the labeled training data.
        :param x_unlabeled: Unlabeled training data.
        :param validation_data: Tuple of validation data and labels.
        :param initial_epochs: Number of epochs for initial training on labeled data.
        :param initial_steps: Number of steps per epoch for initial training.
        :param final_epochs: Number of epochs for training on combined data.
        :param final_steps: Number of steps per epoch for training on combined data.
        :param verbose: Verbosity mode.
        :return: The final trained model and training histories.
        """
        print("Step 1: Training initial model on labeled data...")
        self.initial_model = build_and_compile_cnn_model(dense_units=64)

        labeled_ds = tf.data.Dataset.from_tensor_slices((x_train_lab, y_train_lab)).map(
            lambda x, y: (weak_augment(x), y)).shuffle(1000).repeat().batch(64)

        history_initial = self.initial_model.fit(
            labeled_ds,
            epochs=initial_epochs,
            steps_per_epoch=initial_steps,
            validation_data=validation_data,
            verbose=verbose
        )

        print("\nStep 2: Generating pseudo-labels...")
        x_pseudo, y_pseudo = self._generate_pseudo_labels(x_unlabeled)
        print(f"Generated {len(x_pseudo)} confident pseudo-labels.")

        print("\nStep 3: Retraining on combined data...")
        x_combined = np.concatenate([x_train_lab, x_pseudo], axis=0)
        y_combined = np.concatenate([y_train_lab, y_pseudo], axis=0)

        self.final_model = build_and_compile_cnn_model(dense_units=64)

        combined_ds = tf.data.Dataset.from_tensor_slices((x_combined, y_combined)).map(
            lambda x, y: (weak_augment(x), y)).shuffle(len(x_combined)).repeat().batch(64)

        history_final = self.final_model.fit(
            combined_ds,
            epochs=final_epochs,
            steps_per_epoch=final_steps,
            validation_data=validation_data,
            verbose=verbose
        )

        return self.final_model, history_initial, history_final

    def _generate_pseudo_labels(self, x_unlabeled):
        """
        Generates pseudo-labels for the unlabeled data using the initial model.
        :param x_unlabeled: Unlabeled data.
        :return: Tuple of confident unlabeled data and their pseudo-labels.
        """
        predictions = self.initial_model.predict(x_unlabeled, batch_size=64)
        pseudo_labels = np.argmax(predictions, axis=1)
        max_probabilities = np.max(predictions, axis=1)
        confident_mask = max_probabilities > self.confidence_threshold
        return x_unlabeled[confident_mask], pseudo_labels[confident_mask]
    
    

class FixMatch(Model):
    """
    Implements the FixMatch semi-supervised learning approach.
    """
    
    def __init__(self, model, threshold=0.95, lambda_u=1.0):
        """
        Initializes the FixMatch model.
        :param model: The base model to be used.
        :param threshold: Confidence threshold for pseudo-labeling.
        :param lambda_u: Weight for the unsupervised loss.
        """
        super(FixMatch, self).__init__()
        self.model = model
        self.threshold = threshold
        self.lambda_u = lambda_u
        self.sup_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.unsup_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.sup_loss_tracker = tf.keras.metrics.Mean(name="sup_loss")
        self.unsup_loss_tracker = tf.keras.metrics.Mean(name="unsup_loss")
        self.mask_ratio_tracker = tf.keras.metrics.Mean(name="mask_ratio")
        self.accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    @property
    def metrics(self):
        """
        Returns the list of metrics to be tracked during training.
        """
        return [self.sup_loss_tracker, self.unsup_loss_tracker, self.mask_ratio_tracker, self.accuracy_metric]

    @tf.function
    def train_step(self, data):
        """
        Performs a single training step.
        """
        (x_l, y_l), x_u = data
        with tf.GradientTape() as tape:
            # Generate pseudo-labels with weak augmentation
            x_u_weak = tf.map_fn(weak_augment, x_u)
            logits_u_weak = self.model(x_u_weak, training=False)
            pseudo_labels = tf.argmax(logits_u_weak, axis=1)
            max_probs = tf.reduce_max(tf.nn.softmax(logits_u_weak), axis=1)
            mask = tf.cast(max_probs >= self.threshold, tf.float32)

            # Calculate unsupervised loss on strongly augmented data
            x_u_strong = tf.map_fn(strong_augment, x_u)
            logits_u_strong = self.model(x_u_strong, training=True)
            pseudo_labels_onehot = tf.one_hot(pseudo_labels, depth=10)
            unsup_loss = self.unsup_loss_fn(pseudo_labels_onehot, logits_u_strong, sample_weight=mask)

            # Calculate supervised loss
            x_l_aug = tf.map_fn(weak_augment, x_l)
            logits_l = self.model(x_l_aug, training=True)
            sup_loss = self.sup_loss_fn(y_l, logits_l)

            # Total loss
            total_loss = sup_loss + self.lambda_u * unsup_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.sup_loss_tracker.update_state(sup_loss)
        self.unsup_loss_tracker.update_state(unsup_loss)
        self.mask_ratio_tracker.update_state(tf.reduce_mean(mask))
        self.accuracy_metric.update_state(y_l, logits_l)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        """
        Performs a single testing step.
        """
        x, y = data
        logits = self.model(x, training=False)
        loss = self.sup_loss_fn(y, logits)
        self.accuracy_metric.update_state(y, logits)
        return {"loss": loss, "accuracy": self.accuracy_metric.result()}
    


class UDAModel(Model):
    """
    Implements the UDA (Unsupervised Data Augmentation) semi-supervised learning approach.
    """
    
    def __init__(self, model, threshold=0.95, lambda_u=1.0):
        """
        Initializes the UDA model.
        """
        super(UDAModel, self).__init__()
        self.model = model
        self.threshold = threshold
        self.lambda_u = lambda_u
        self.sup_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.cons_loss_fn = tf.keras.losses.KLDivergence()
        self.sup_loss_tracker = tf.keras.metrics.Mean(name="sup_loss")
        self.cons_loss_tracker = tf.keras.metrics.Mean(name="cons_loss")
        self.mask_ratio_tracker = tf.keras.metrics.Mean(name="mask_ratio")
        self.accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    @property
    def metrics(self):
        """
        Returns the list of metrics to be tracked during training.
        """
        return [self.sup_loss_tracker, self.cons_loss_tracker, self.mask_ratio_tracker, self.accuracy_metric]

    @tf.function
    def train_step(self, data):
        """
        Performs a single training step.
        """
        (x_l, y_l), x_u = data
        with tf.GradientTape() as tape:
            # Supervised loss
            x_l_aug = tf.map_fn(weak_augment, x_l)
            logits_l = self.model(x_l_aug, training=True)
            sup_loss = self.sup_loss_fn(y_l, logits_l)

            # Consistency loss (original vs. strongly augmented)
            logits_u_orig = self.model(x_u, training=False)
            # CORRECTED: Using strong_augment for consistency as per the UDA paper's principle
            logits_u_aug = self.model(tf.map_fn(strong_augment, x_u), training=True)
            probs_u_orig = tf.nn.softmax(logits_u_orig)
            probs_u_aug = tf.nn.softmax(logits_u_aug)

            # Confidence-based masking
            max_probs = tf.reduce_max(probs_u_orig, axis=1)
            mask = tf.cast(max_probs >= self.threshold, tf.float32)

            cons_loss = self.cons_loss_fn(probs_u_orig, probs_u_aug, sample_weight=mask)

            # Total loss
            total_loss = sup_loss + self.lambda_u * cons_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.sup_loss_tracker.update_state(sup_loss)
        self.cons_loss_tracker.update_state(cons_loss)
        self.mask_ratio_tracker.update_state(tf.reduce_mean(mask))
        self.accuracy_metric.update_state(y_l, logits_l)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        """
        Performs a single testing step.
        """
        x, y = data
        logits = self.model(x, training=False)
        loss = self.sup_loss_fn(y, logits)
        self.accuracy_metric.update_state(y, logits)
        return {"loss": loss, "accuracy": self.accuracy_metric.result()}