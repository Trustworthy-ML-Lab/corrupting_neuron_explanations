import logging

import torch
import torch.nn as nn

from util.common import get_threshold


class SoftIoU(nn.Module):
    def __init__(self, model, layer, unit, a=1):
        super(SoftIoU, self).__init__()

        # create logger
        self.logger = logging.getLogger(__name__)

        # setup constants
        self.cutoff = get_threshold(model, layer)[unit]
        self.a = a

        # log constants
        self.logger.info(f"Using cutoff {self.cutoff}, a = {self.a}")

    def forward(self, input, target):
        """
        Returns loss in the overlap between input and target
        """
        # input: (batch_size, 1, H, W), features of interest
        # target: (batch_size, 1, H, W)
        # a: scaling coeff, higher a -> sharper m_hat

        m_hat = torch.sigmoid(self.a * (input - self.cutoff) / torch.std(input))
        soft_intersection = (m_hat * target).sum()
        soft_union = torch.max(m_hat, target).sum()
        loss = soft_intersection / (soft_union + 1e-8)  # scalar number

        return loss
