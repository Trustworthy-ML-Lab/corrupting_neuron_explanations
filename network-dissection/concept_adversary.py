import logging

import torch
import torch.nn.functional as F

from loss.dissection_loss import DissectionLoss
from loss.soft_iou_loss import SoftIoU
from util.common import get_normalization_layer


class FGSMConceptAdversary:
    def __init__(
        self,
        model,
        model_name,
        model_type,
        layer,
        unit,
        loss_type,
        epsilon,
    ):
        self.model = model
        self.layer = layer
        self.unit = unit
        self.epsilon = epsilon

        # set up the device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # setup loss
        if loss_type == "basic":
            self.loss = DissectionLoss()
        elif loss_type == "soft_iou":
            self.loss = SoftIoU(model_name, layer, unit)
        else:
            raise ValueError("Unknown loss type: {}".format(loss_type))

        # setup normalization layer
        self.norm = get_normalization_layer(model_type)

        # log eps and steps
        logger = logging.getLogger(__name__)
        logger.info(
            f"Running FGSMConceptAdversary attack with loss {loss_type}, epsilon {self.epsilon}"
        )

    def attack(
        self,
        images,
        source_map,
        target_map,
        iw=112,
        ih=112,
    ):
        """
        Implement PGD attack
        Source: https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgd.py
        """
        # set into evaluation mode
        self.model.eval()

        # convert images to tensor
        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images).to(
                self.device
            )  # [batch_size, 1, H_prime, W_prime]

        if not isinstance(source_map, torch.Tensor):
            source_map = torch.from_numpy(source_map).to(
                self.device
            )  # [batch_size, 1, H, W]

        if not isinstance(target_map, torch.Tensor):
            target_map = torch.from_numpy(target_map).to(
                self.device
            )  # [batch_size, 1, H, W]

        # run pgd attack
        adv_images = images.clone().detach()
        adv_images.requires_grad = True

        # run model, extract channel `self.unit` and upsample to map size
        outputs = self.model(self.norm(adv_images))
        outputs = outputs[
            :, self.unit : self.unit + 1, :, :
        ]  # [batch_size, 1, H_c, W_c]
        outputs = F.interpolate(
            outputs, size=(ih, iw), mode="bilinear"
        )  # [batch_size, 1, H, W]

        # Calculate loss
        cost = 0.0
        cost += self.loss(outputs, source_map)
        cost -= self.loss(outputs, target_map)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, adv_images, retain_graph=False, create_graph=False
        )[0]

        adv_images = adv_images - self.epsilon * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images


class PGDConceptAdversary:
    def __init__(
        self,
        model,
        model_name,
        model_type,
        layer,
        unit,
        loss_type,
        epsilon=(4.0 / 255.0),
        steps=50,
    ):
        self.model = model
        self.layer = layer
        self.unit = unit
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = (self.epsilon / steps) * 2

        # set up the device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # setup loss
        if loss_type == "basic":
            self.loss = DissectionLoss()
        elif loss_type == "soft_iou":
            self.loss = SoftIoU(model_name, layer, unit)
        else:
            raise ValueError("Unknown loss type: {}".format(loss_type))

        # setup normalization layer
        self.norm = get_normalization_layer(model_type)

        # log eps and steps
        logger = logging.getLogger(__name__)
        logger.info(
            f"Running PGDConceptAdversary attack with loss {loss_type}, epsilon {self.epsilon}, alpha {self.alpha}, and {self.steps} steps"
        )

    def attack(
        self,
        images,
        source_map,
        target_map,
        iw=112,
        ih=112,
    ):
        """
        Implement PGD attack
        Source: https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgd.py
        """
        # set into evaluation mode
        self.model.eval()

        # convert images to tensor
        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images).to(
                self.device
            )  # [batch_size, 1, H_prime, W_prime]

        if not isinstance(source_map, torch.Tensor):
            source_map = torch.from_numpy(source_map).to(
                self.device
            )  # [batch_size, 1, H, W]

        if not isinstance(target_map, torch.Tensor):
            target_map = torch.from_numpy(target_map).to(
                self.device
            )  # [batch_size, 1, H, W]

        # run pgd attack
        adv_images = images.clone().detach()
        adv_images = adv_images + torch.empty_like(adv_images).zero_()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True

            # run model, extract channel `self.unit` and upsample to map size
            outputs = self.model(self.norm(adv_images))
            outputs = outputs[
                :, self.unit : self.unit + 1, :, :
            ]  # [batch_size, 1, H_c, W_c]
            outputs = F.interpolate(
                outputs, size=(ih, iw), mode="bilinear"
            )  # [batch_size, 1, H, W]

            # Calculate loss
            cost = 0.0
            cost += self.loss(outputs, source_map)
            cost -= self.loss(outputs, target_map)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() - self.alpha * grad.sign()
            delta = torch.clamp(
                adv_images - images, min=-self.epsilon, max=self.epsilon
            )
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
