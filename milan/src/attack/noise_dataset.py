import logging
from typing import Optional

import numpy as np
import torch
import torch.utils.data
import torchvision


def get_uniform_noise(img_tensor_shape, noise_std):
    uniform_rv = torch.rand(img_tensor_shape)
    noise = (2.0 * uniform_rv - 1.0) * noise_std
    return noise


def get_gaussian_noise(img_tensor_shape, noise_std):
    noise = torch.randn(img_tensor_shape) * noise_std
    return noise


def get_bernoulli_noise(img_tensor_shape, noise_std):
    bernoulli_rv = torch.bernoulli(0.5 * torch.ones(img_tensor_shape))
    noise = (2.0 * bernoulli_rv - 1.0) * noise_std
    return noise


class NoiseDataset(torch.utils.data.Dataset):
    """
    Return batches with source and target masks
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        noise_type: str,
        noise_std: float,
        attack_probability: float,
        transform_after_attack: Optional[torchvision.transforms.Compose] = None,
        **kwargs,
    ):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        # set up attack variables
        self.dataset = dataset
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.kwargs = kwargs
        self.transform_after_attack = transform_after_attack
        if transform_after_attack is not None:
            # compose the dataset transform with the transform after attack
            self.transform = torchvision.transforms.Compose(
                self.dataset.transform.transforms + transform_after_attack.transforms
            )
            self.logger.info(f"Combined transform: {self.transform}")
        else:
            self.transform = self.dataset.transform
            self.logger.info(f"Transform: {self.transform}")

        self.attack_idx = np.random.choice(
            np.arange(len(self.dataset)),
            int(len(self.dataset) * attack_probability),
            replace=False,
        )

        # assert unique attack idx
        assert len(self.attack_idx) == len(set(self.attack_idx))

        self.logger.info(
            f"Attack probability: {attack_probability}, attack idx length: {len(self.attack_idx)}/{len(self.dataset)}, noise type: {noise_type}, noise std: {noise_std}"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, label = self.dataset[idx]

        # add noise to image
        if self.noise_type == "gaussian":
            # create noise with dtype of image
            noise = get_gaussian_noise(img.shape, self.noise_std)
        elif self.noise_type == "uniform":
            # create noise with dtype of image
            noise = get_uniform_noise(img.shape, self.noise_std)
        elif self.noise_type == "bernoulli":
            # create noise with dtype of image
            noise = get_bernoulli_noise(img.shape, self.noise_std)
        else:
            raise ValueError(f"Noise type {self.noise_type} not supported")

        # check if idx is in attack_idx
        if idx in self.attack_idx:
            img = img + noise
            img = torch.clamp(img, 0, 1)

        return self.transform_after_attack(img), label
