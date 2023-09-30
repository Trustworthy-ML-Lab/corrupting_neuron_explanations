import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.utils.data
import torchvision

from src.utils.common import get_unit_info


class AttackDataset(torch.utils.data.Dataset):
    """
    Return batches with source and target masks
    """

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        dataset: torch.utils.data.Dataset,
        result_dir: Path,
        source_layer: str,
        source_unit: int,
        attack_image_count: int,
        target_layer: Optional[str] = None,
        target_unit: Optional[int] = None,
        transform_after_attack: Optional[torchvision.transforms.Compose] = None,
    ):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        # set up attack variables
        self.dataset = dataset
        self.source_layer = source_layer
        self.source_unit = source_unit
        self.target_layer = target_layer
        self.target_unit = target_unit
        if transform_after_attack is not None:
            # compose the dataset transform with the transform after attack
            self.transform = torchvision.transforms.Compose(
                self.dataset.transform.transforms + transform_after_attack.transforms
            )
            self.logger.info(f"Combined transform: {self.transform}")
        else:
            self.transform = self.dataset.transform
            self.logger.info(f"Transform: {self.transform}")

        # get info on units to modify
        self.source_attack_info = get_unit_info(
            result_dir, model_name, dataset_name, source_layer, source_unit
        )
        self.source_mask_shape = self.source_attack_info["mask"][0].shape
        assert (
            self.source_mask_shape[0] == 1
        ), "Source mask shape should be (1, length of ids, 1, width, height)"

        self.source_attack_ids = self.source_attack_info["id"][
            :attack_image_count
        ]  # (length of ids,)
        self.source_attack_masks = self.source_attack_info["mask"][
            :attack_image_count, :, :, :
        ]  # (length of ids, 1, 224, 224)
        self.logger.info(
            f"source_mask_shape: {self.source_mask_shape}, source_attack_ids shape: {len(self.source_attack_ids)}, source_attack_masks shape: {len(self.source_attack_masks)}"
        )

        if target_unit is not None:
            self.target_attack_info = get_unit_info(
                result_dir, model_name, dataset_name, target_layer, target_unit
            )
            self.target_attack_ids = self.target_attack_info["id"][
                :attack_image_count
            ]  # (length of ids,)
            self.target_attack_masks = self.target_attack_info["mask"][
                :attack_image_count, :, :, :
            ]  # (length of ids, 1, 224, 224)
            assert self.target_attack_info["mask"][0].shape == self.source_mask_shape
            self.logger.info(
                f"target_mask_shape: {self.source_mask_shape}, target_attack_ids shape: {len(self.target_attack_ids)}, target_attack_masks shape: {len(self.target_attack_masks)}"
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, label = self.dataset[idx]

        # return if source unit is not specified and target unit is not specified
        if self.source_unit is None and self.target_unit is None:
            return img, label

        source_mask = np.zeros(self.source_mask_shape, dtype=np.uint8)
        if idx in self.source_attack_ids:
            idx_idx = self.source_attack_ids.index(idx)
            source_mask = self.source_attack_masks[idx_idx]  # (1, 224, 224)
            self.logger.debug(f"Found source mask for idx {idx} at idx_idx {idx_idx}")

        target_mask = np.zeros(self.source_mask_shape, dtype=np.uint8)
        if self.target_unit is not None and idx in self.target_attack_ids:
            idx_idx = self.target_attack_ids.index(idx)
            target_mask = self.target_attack_masks[idx_idx]  # (1, 224, 224)
            self.logger.debug(f"Found target mask for idx {idx} at idx_idx {idx_idx}")

        return img, source_mask, target_mask, label
