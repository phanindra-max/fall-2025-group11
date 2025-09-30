# -*- coding: utf-8 -*-
"""
Author: Prudhvi Chekuri
Date: 2025-09-28
Version: 1.0
"""

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - accuracy: computes the accuracy over the k top predictions for the specified values of k.
    - AverageMeter: computes and stores the average and current value.
'''
import logging

import torch

logger = logging.getLogger(__name__)

__all__ = ['get_mean_and_std', 'accuracy', 'AverageMeter']


def get_mean_and_std(dataset):
    '''
    Computes the mean and standard deviation of a dataset.

    Note:
        This function assumes the dataset returns tensors for 3-channel images.
        For single-channel images, it might need modification.

    Args:
        dataset (torch.utils.data.Dataset): The dataset for which to compute the stats.

    Returns:
        tuple: A tuple containing two tensors:
            - mean (torch.Tensor): The mean of the dataset across each channel.
            - std (torch.Tensor): The standard deviation of the dataset across each channel.
    '''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k.

    Args:
        output (torch.Tensor): The model's output logits. Shape: (batch_size, num_classes).
        target (torch.Tensor): The ground truth labels. Shape: (batch_size).
        topk (tuple, optional): A tuple of integers specifying the values of 'k'
            for which to compute the top-k accuracy. Defaults to (1,).

    Returns:
        list: A list of floats, where each float is the top-k accuracy in
              percentage for the corresponding 'k' in the topk tuple.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """
    Computes and stores the average and current value of a metric.

    This is a utility class for tracking metrics like loss or accuracy over time.

    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        """Initializes the AverageMeter object."""
        self.reset()

    def reset(self):
        """Resets all the statistics to their initial values."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter's statistics with a new value.

        Args:
            val (float or int): The new value to add.
            n (int, optional): The number of samples associated with 'val'.
                               Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
