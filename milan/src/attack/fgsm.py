from typing import Any

import torch
import torch.nn as nn
import torch.utils.data
import torchvision

from src.utils import logging
from src.utils.common import shrink_to_layer


class FGSM:
    def __init__(
        self,
        eps: float,
        loss: nn.Module,
        model: str,
        transform: Any,
        layer: str,
        source_unit: int,
    ):
        self.eps = eps
        self.loss = loss
        self.model = model
        self.layer = layer
        self.source_unit = source_unit

        # combine list of transforms in a single transform
        if isinstance(transform, list):
            self.transform = torchvision.transforms.Compose(transform)
        else:
            self.transform = transform

        # log attack parameters
        self.logger = logging.get_logger(self.__class__.__name__)
        self.logger.info(
            f"FGSM attack with eps={self.eps}, loss={self.loss}, model={self.model}, transform={self.transform}, layer={self.layer}"
        )

    def __call__(
        self,
        images: torch.Tensor,
        source_maps: torch.Tensor,
        target_maps: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        img: torch.tensor (C, H, W)
        source_map: torch.Tensor (1, H, W)
        target_map: torch.Tensor (1, H, W)
        """
        adv_images = images.clone().detach()
        adv_images.requires_grad = True

        # run model, extract channel `self.unit` and upsample to map size
        outputs = self.model(self.transform(adv_images))
        outputs = outputs[
            :, self.source_unit : self.source_unit + 1, :, :
        ]  # (1, 1, H, W)
        outputs = torch.nn.functional.interpolate(
            outputs, size=source_maps.shape[2:], mode="bilinear", align_corners=False
        )

        # Calculate loss
        cost = 0.0
        cost += self.loss(outputs, source_maps)
        cost -= self.loss(outputs, target_maps)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, adv_images, retain_graph=False, create_graph=False
        )[0]

        adv_images = adv_images - self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return [self.transform(adv_images), labels]
