import torch
import torch.nn as nn
import torch.nn.functional as F


class DissectionLoss(nn.Module):
    def __init__(self):
        super(DissectionLoss, self).__init__()

    def forward(self, input, target):
        """
        Returns loss in the overlap between input and target
        """
        # calculate the overlap between input and target
        overlap = input * target  # (batch_size, 3, H, W)
        overlap = overlap.sum()  # scalar number
        overlap = overlap / (target.sum() + 1e-8)  # scalar number
        return overlap
