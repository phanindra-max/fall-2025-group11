# -*- coding: utf-8 -*-
"""
Author: Prudhvi Chekuri
Date: 2025-09-27
Version: 1.0
"""

import logging
import math
import os

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
mnist_mean = (0.1307,)
mnist_std = (0.3081,)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(args, root):
    """
    Prepares the CIFAR-10 dataset for semi-supervised learning.

    This function splits the CIFAR-10 training set into labeled and unlabeled
    subsets and applies appropriate transformations for training and validation.

    Args:
        args (Namespace): A namespace object containing command-line arguments,
                          such as the number of labeled examples.
        root (str): The root directory where the dataset is stored.

    Returns:
        tuple: A tuple containing:
            - train_labeled_dataset (Dataset): The labeled training dataset.
            - train_unlabeled_dataset (Dataset): The unlabeled training dataset.
            - test_dataset (Dataset): The test dataset.
    """
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                             padding=int(32*0.125),
                             padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    """
    Splits the dataset indices into labeled and unlabeled sets.

    Args:
        args (Namespace): A namespace object containing arguments like
                          num_labeled, num_classes, etc.
        labels (list or np.ndarray): A list of labels for the entire dataset.

    Returns:
        tuple: A tuple containing:
            - labeled_idx (np.ndarray): An array of indices for the labeled data.
            - unlabeled_idx (np.ndarray): An array of indices for the unlabeled data.
    """
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    """
    A transformation class that applies weak and strong augmentations,
    as required by the FixMatch algorithm.
    """
    def __init__(self, mean, std, dataset='cifar10'):
        """
        Initializes the weak and strong augmentation pipelines.

        Args:
            mean (tuple or list): The mean values for normalization.
            std (tuple or list): The standard deviation values for normalization.
            dataset (str, optional): The name of the dataset ('cifar10' or 'mnist').
                                     Defaults to 'cifar10'.
        """
        self.weak = transforms.Compose([
            transforms.Pad(2) if dataset == 'mnist' else transforms.Lambda(lambda x: x),  # Pad for MNIST
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.Pad(2) if dataset == 'mnist' else transforms.Lambda(lambda x: x),  # Pad for MNIST
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        """
        Applies the transformations to an input image.

        Args:
            x (PIL.Image): The input image.

        Returns:
            tuple: A tuple containing the weakly and strongly augmented images,
                   both normalized.
        """
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    """
    A custom CIFAR-10 dataset for semi-supervised learning (SSL).

    This class extends torchvision's CIFAR10 dataset to only include
    a subset of the data specified by indices.
    """
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        """
        Initializes the CIFAR10SSL dataset.

        Args:
            root (str): Root directory of the dataset.
            indexs (list or np.ndarray): A list of indices to include in the dataset.
            train (bool, optional): If True, creates dataset from training set,
                                    otherwise creates from test set. Defaults to True.
            transform (callable, optional): A function/transform to apply to the images.
                                            Defaults to None.
            target_transform (callable, optional): A function/transform to apply to the labels.
                                                   Defaults to None.
            download (bool, optional): If true, downloads the dataset from the internet.
                                       Defaults to False.
        """
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset at a specific index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple (image, target) where target is the class index.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_mnist(args, root):
    """
    Prepares the MNIST dataset for semi-supervised learning.

    This function splits the MNIST training set into labeled and unlabeled
    subsets and applies appropriate transformations for training and validation.

    Args:
        args (Namespace): A namespace object containing command-line arguments.
        root (str): The root directory where the dataset is stored.

    Returns:
        tuple: A tuple containing:
            - train_labeled_dataset (Dataset): The labeled training dataset.
            - train_unlabeled_dataset (Dataset): The unlabeled training dataset.
            - test_dataset (Dataset): The test dataset.
    """
    transform_labeled = transforms.Compose([
        transforms.Pad(2),  # Pad 28x28 to 32x32
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                             padding=int(32*0.125),
                             padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=mnist_mean, std=mnist_std)
    ])
    transform_val = transforms.Compose([
        transforms.Pad(2),  # Pad 28x28 to 32x32
        transforms.ToTensor(),
        transforms.Normalize(mean=mnist_mean, std=mnist_std)
    ])
    base_dataset = datasets.MNIST(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = MNISTSSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = MNISTSSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=mnist_mean, std=mnist_std, dataset='mnist'))

    test_dataset = datasets.MNIST(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class MNISTSSL(datasets.MNIST):
    """
    A custom MNIST dataset for semi-supervised learning (SSL).

    This class extends torchvision's MNIST dataset to only include a subset
    of the data specified by indices. It also overrides folder properties
    to prevent "dataset not found" errors in certain environments.
    """
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        """
        Initializes the MNISTSSL dataset.

        Args:
            root (str): Root directory of the dataset.
            indexs (list or np.ndarray): A list of indices to include in the dataset.
            train (bool, optional): If True, creates dataset from training set. Defaults to True.
            transform (callable, optional): A function/transform for the images. Defaults to None.
            target_transform (callable, optional): A function/transform for the labels. Defaults to None.
            download (bool, optional): If true, downloads the dataset. Defaults to False.
        """
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    @property
    def raw_folder(self) -> str:
        """Returns the path to the raw data folder."""
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self) -> str:
        """Returns the path to the processed data folder."""
        return os.path.join(self.root, 'MNIST', 'processed')

    def _check_exists(self) -> bool:
        """Checks if the dataset files exist in the raw folder."""
        exists = all(
            os.path.exists(os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0]))
            for url, _ in self.resources
        )
        logger.debug(f"Checking dataset existence in {self.raw_folder}: {exists}")
        return exists

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset at a specific index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple (image, target) where target is the class index.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(), mode='L')  # Convert to PIL Image (grayscale)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


DATASET_GETTERS = {'cifar10': get_cifar10, 'mnist': get_mnist}