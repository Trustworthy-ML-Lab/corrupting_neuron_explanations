import csv
import glob
from collections import OrderedDict
from os import remove
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch.nn as nn
import torch.utils.data
import torchvision

from src import milannotations

from . import logging

MODEL_TANSFORMS = {}
MODEL_TANSFORMS[milannotations.KEYS.CIFAR10] = torchvision.transforms.Compose(
    [
        torchvision.transforms.Normalize(
            (0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)
        )
    ]
)


logger = logging.get_logger(__name__)


def get_unit_info(
    root: Path, model: str, dataset: str, layer: str, unit: int
) -> Dict[str, Any]:
    info = {}
    unit_idx = unit

    # get image ids
    image_ids = None

    # first check unit specific results folder exists
    result_dir_glob = root / "baseline" / model / dataset / layer / str(unit) / "*"
    if len(glob.glob(str(result_dir_glob))) == 0:
        logger.info(f"no results found for {result_dir_glob}")
        result_dir_glob = root / "baseline" / model / dataset / layer / "all" / "*"
    else:
        unit_idx = 0
        logger.info(f"results found for {result_dir_glob}")

    # resolve glob
    result_dir = None
    matched_dirs = glob.glob(str(result_dir_glob))
    matched_dirs.sort(reverse=True)
    for rd in matched_dirs:
        result_dir = rd
        result_dir = Path(result_dir)
        logger.info(f"result_dir: {result_dir}")
        break

    image_ids_file = result_dir / "output" / "ids.csv"
    with open(image_ids_file, "r") as f:
        reader = csv.reader(f)
        # read comma seperated values from `unit` row number
        for idx, row in enumerate(reader):
            if idx == unit_idx:
                image_ids = row
                break

    assert image_ids is not None
    info["id"] = [int(image_id) for image_id in image_ids]

    # get image masks
    image_masks_file = result_dir / "output" / "masks.npy"
    image_masks = np.load(image_masks_file)[unit_idx]  # (num_images, 1, H, W)
    assert len(image_ids) == image_masks.shape[0]
    info["mask"] = image_masks

    # get description
    # csv file format ("layer", "unit", "description")
    # description = None
    # description_file = result_dir / "descriptions.csv"
    # with open(description_file, "r") as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         if int(row[1]) == unit:
    #             description = row[2]
    #             break

    # assert description is not None
    # info["description"] = description
    logger.debug(f"info: {info}")

    return info


def shrink_to_layer(model: Any, layer: str):
    """
    Shrink model to the layer
    FOR EXAMPLE, if model is a ResNet50 and layer_name is "layer4.2.conv3",
    then the model will be shrinked to the layer4.2.conv3
    """
    # get layer name
    first_part = layer.split(".")[0]
    last_part = layer.split(".")[1:]

    # get layer
    layers = OrderedDict()

    # handle case for DataParallel model
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    for key, value in model._modules.items():
        if key == first_part:
            if len(last_part) > 0:
                layers[key] = shrink_to_layer(value, ".".join(last_part))
            else:
                layers[key] = value

            return nn.Sequential(layers)
        else:
            layers[key] = value

    raise ValueError(f"Layer {layer} not found in model")


def remove_transforms_for_adversarial_attack(
    dataset: torch.utils.data.Dataset, logger: Optional[Any]
) -> Tuple[torch.utils.data.Dataset, List[Any]]:
    """
    Remove all transforms except for to_tensor and resize for adversarial attack
    """
    # get transforms
    keep_transforms = []
    remove_transforms = []

    for transform in dataset.transform.transforms:
        if isinstance(transform, torchvision.transforms.ToTensor):
            keep_transforms.append(transform)
        elif isinstance(transform, torchvision.transforms.Resize):
            keep_transforms.append(transform)
        elif isinstance(transform, torchvision.transforms.CenterCrop):
            keep_transforms.append(transform)
        else:
            remove_transforms.append(transform)

    # compose transforms
    keep_transform = torchvision.transforms.Compose(keep_transforms)
    remove_transform = torchvision.transforms.Compose(remove_transforms)

    # remove transforms
    dataset.transform = keep_transform

    if logger is not None:
        logger.info(
            f"keep transforms: {keep_transform}, remove transforms: {remove_transform}"
        )

    return dataset, remove_transform
