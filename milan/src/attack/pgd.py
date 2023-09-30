from typing import Any

import torch
import torch.nn as nn
import torch.utils.data
import torchvision

from src.utils import logging


class PGD:
    def __init__(
        self,
        eps: float,
        steps: int,
        loss: nn.Module,
        model: str,
        transform: Any,
        layer: str,
        source_unit: int,
    ):
        self.eps = eps
        self.steps = steps
        self.loss = loss
        self.model = model
        self.layer = layer
        self.source_unit = source_unit
        self.alpha = (2.0 * eps) / steps

        # combine list of transforms in a single transform
        if isinstance(transform, list):
            self.transform = torchvision.transforms.Compose(transform)
        else:
            self.transform = transform

        # log attack parameters
        self.logger = logging.get_logger(self.__class__.__name__)
        self.logger.info(
            f"FGSM attack with eps={self.eps}, alpha={self.alpha}, steps={self.steps}, loss={self.loss}, model={self.model}, transform={self.transform}, layer={self.layer}, source_unit={self.source_unit}"
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

        # run pgd attack
        adv_images = images.clone().detach()
        adv_images = adv_images + torch.empty_like(adv_images).zero_()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True

            # run model, extract channel `self.unit` and upsample to map size
            outputs = self.model(self.transform(adv_images))
            outputs = outputs[
                :, self.source_unit : self.source_unit + 1, :, :
            ]  # (1, 1, H, W)
            outputs = torch.nn.functional.interpolate(
                outputs,
                size=source_maps.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

            # Calculate loss
            cost = 0.0
            cost += self.loss(outputs, source_maps)
            cost -= self.loss(outputs, target_maps)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() - self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return [self.transform(adv_images), labels]
