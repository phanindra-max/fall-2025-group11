# -*- coding: utf-8 -*-
"""
Author: Prudhvi Chekuri
Date: 2025-09-28
Version: 1.0
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def mish(x):
    """
    Applies the Mish activation function.

    Mish is a self-regularized, non-monotonic activation function defined as:
    f(x) = x * tanh(softplus(x))

    Reference: "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
    (https://arxiv.org/abs/1908.08681)

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The output tensor after applying Mish activation.
    """
    return x * torch.tanh(F.softplus(x))


class PSBatchNorm2d(nn.BatchNorm2d):
    """
    Implements Parametric-Shift Batch Normalization.

    This variant of BatchNorm adds a small learnable parameter 'alpha' to the
    output, which is intended to help prevent filter collapse in neural networks.

    Reference: "How Does BN Increase Collapsed Neural Network Filters?"
    (https://arxiv.org/abs/2001.11216)
    """

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        """
        Initializes the PSBatchNorm2d layer.

        Args:
            num_features (int): The number of features in the input.
            alpha (float, optional): The parametric shift value. Defaults to 0.1.
            eps (float, optional): A value added to the denominator for numerical stability. Defaults to 1e-05.
            momentum (float, optional): The value used for the running_mean and running_var computation. Defaults to 0.001.
            affine (bool, optional): If True, this module has learnable affine parameters. Defaults to True.
            track_running_stats (bool, optional): If True, this module tracks the running mean and variance. Defaults to True.
        """
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        """
        Performs the forward pass of the PSBatchNorm2d layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor with the parametric shift applied.
        """
        return super().forward(x) + self.alpha


class BasicBlock(nn.Module):
    """
    A basic residual block for WideResNet.

    This block consists of two 3x3 convolutional layers, each preceded by
    Batch Normalization and a LeakyReLU activation function. It includes a
    skip connection (residual connection).
    """
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        """
        Initializes the BasicBlock.

        Args:
            in_planes (int): Number of input channels.
            out_planes (int): Number of output channels.
            stride (int): The stride for the first convolutional layer.
            drop_rate (float, optional): Dropout rate. Defaults to 0.0.
            activate_before_residual (bool, optional): If True, applies activation
                to the input before the residual connection. Defaults to False.
        """
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                 padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        """
        Performs the forward pass through the BasicBlock.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor of the block.
        """
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    """
    A block of sequential BasicBlocks that forms a stage in the WideResNet.
    """
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        """
        Initializes the NetworkBlock.

        Args:
            nb_layers (int): The number of BasicBlocks in this network block.
            in_planes (int): The number of input channels.
            out_planes (int): The number of output channels.
            block (nn.Module): The type of block to use (e.g., BasicBlock).
            stride (int): The stride for the first block in this sequence.
            drop_rate (float, optional): Dropout rate. Defaults to 0.0.
            activate_before_residual (bool, optional): Activation setting for the blocks. Defaults to False.
        """
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        """
        Creates a sequence of BasicBlocks.

        Args:
            block (nn.Module): The block class to instantiate.
            in_planes (int): Input channels for the first block.
            out_planes (int): Output channels for all blocks.
            nb_layers (int): The number of blocks to create.
            stride (int): Stride for the very first block in the sequence.
            drop_rate (float): Dropout rate for all blocks.
            activate_before_residual (bool): Activation setting.

        Returns:
            nn.Sequential: A sequential container of the created blocks.
        """
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Performs the forward pass through the NetworkBlock.
        """
        return self.layer(x)


class WideResNet(nn.Module):
    """
    Implementation of a Wide Residual Network.
    """
    def __init__(self, num_classes, depth=28, widen_factor=2, drop_rate=0.0, dataset='cifar10'):
        """
        Initializes the WideResNet model.

        Args:
            num_classes (int): The number of output classes.
            depth (int, optional): The total depth of the network. Defaults to 28.
            widen_factor (int, optional): The factor by which to widen the channels. Defaults to 2.
            drop_rate (float, optional): The dropout rate. Defaults to 0.0.
            dataset (str, optional): The dataset being used ('cifar10' or 'mnist')
                to adjust the first layer's input channels. Defaults to 'cifar10'.
        """
        super(WideResNet, self).__init__()
        channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(1 if dataset == 'mnist' else 3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)  # Handle 1 channel for MNIST
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, 1, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(channels[3], num_classes)
        self.channels = channels[3]

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                      mode='fan_out',
                                      nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        Performs the forward pass through the WideResNet.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output logits from the final fully connected layer.
        """
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        return self.fc(out)


def build_wideresnet(depth, widen_factor, dropout, num_classes, dataset='cifar10'):
    """
    Factory function to build and return a WideResNet model.

    Args:
        depth (int): The total depth of the network.
        widen_factor (int): The widening factor for the channels.
        dropout (float): The dropout rate.
        num_classes (int): The number of output classes.
        dataset (str, optional): The dataset name ('cifar10' or 'mnist').
            Defaults to 'cifar10'.

    Returns:
        WideResNet: An instance of the WideResNet model.
    """
    logger.info(f"Model: WideResNet {depth}x{widen_factor}")
    return WideResNet(depth=depth,
                      widen_factor=widen_factor,
                      drop_rate=dropout,
                      num_classes=num_classes,
                      dataset=dataset)
