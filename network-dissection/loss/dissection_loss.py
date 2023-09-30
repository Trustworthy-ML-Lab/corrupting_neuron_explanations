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
        # input: (batch_size, 3, H, W)
        # target: (batch_size, H, W)

        # image_height = input.size(2)
        # image_width = input.size(3)

        # calculate the overlap between input and target
        overlap = input * target  # (batch_size, 3, H, W)
        overlap = overlap.sum()  # scalar number
        overlap = overlap / (target.sum() + 1e-8)  # scalar number

        # input = F.relu(input - threshold) / threshold

        # # clamp input between 0 and 1
        # input = torch.clamp(input, min=0, max=1)

        # # calculate mse loss between input and target
        # loss = 1.0 - (torch.mean((input - target) ** 2) / (image_height * image_width))

        return overlap
