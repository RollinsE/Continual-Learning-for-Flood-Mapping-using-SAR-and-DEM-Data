import logging

import numpy as np
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        num_parameters = sum(np.prod(p.size()) for p in model_parameters)
        self.logger.info('Number of trainable parameters: %d', num_parameters)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        num_parameters = sum(np.prod(p.size()) for p in model_parameters)
        return super(BaseModel, self).__str__() + f'\nNumber of trainable parameters: {num_parameters}'
