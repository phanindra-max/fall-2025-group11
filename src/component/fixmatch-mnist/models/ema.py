# -*- coding: utf-8 -*-
"""
Author: Prudhvi Chekuri
Date: 2025-09-28
Version: 1.0
"""

from copy import deepcopy

import torch


class ModelEMA(object):
    """
    Implements Exponential Moving Average for a model's parameters.

    Model EMA is a technique used to stabilize the training process and improve
    the generalization of the final model. It maintains a shadow copy of the
    model's weights that is updated more slowly than the main model's weights.
    This is achieved by taking a weighted average of the current model's
    weights and the previous shadow weights.

    Usage:
        ema = ModelEMA(args, model, decay=0.999)
        ...
        # In the training loop, after an optimizer step
        ema.update(model)
    """
    def __init__(self, args, model, decay):
        """
        Initializes the ModelEMA object.

        Args:
            args (Namespace): A namespace object containing command-line arguments,
                              including the device to use.
            model (torch.nn.Module): The model to which EMA will be applied.
            decay (float): The decay factor for the moving average. A higher
                           value means the old weights have more influence.
        """
        self.ema = deepcopy(model)
        self.ema.to(args.device)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """
        Updates the EMA model's weights.

        This method should be called after each training step (optimizer.step()).
        It updates the parameters of the EMA model using the formula:
        `ema_param = decay * ema_param + (1 - decay) * model_param`

        Buffers (like running means in batch normalization) are copied directly
        from the trained model without averaging.

        Args:
            model (torch.nn.Module): The current trained model from which to
                                     update the EMA weights.
        """
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])
