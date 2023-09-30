import os
import time
from multiprocessing.pool import ThreadPool

import numpy as np
import torch
import torchattacks
from torchattacks.attack import Attack

from loader.data_loader import (
    SegmentationData,
    SegmentationPrefetcher,
    get_noise_attack_data_directory,
)
from util.common import save_image_from_tensor


class UniformNoise(Attack):
    def __init__(self, model, std):
        self.std = std
        super().__init__("UniformNoise", model)

    def forward(self, images):
        # create uniform noise attack
        uniform_rv = torch.rand(images.shape).to(images.device)
        rv = -self.std + 2.0 * uniform_rv * self.std
        adv_images = images + rv
        adv_images = torch.clamp(adv_images, min=0, max=1)
        return adv_images


class BernoulliNoise(Attack):
    def __init__(self, model, std, p=0.5):
        self.std = std
        self.p = p
        super().__init__("BernoulliNoise", model)

    def forward(self, images):
        # create bernoulli rv with self.p success rate
        p_array = self.p * torch.ones(images.shape).to(images.device)
        bernoulli_rv = torch.bernoulli(p_array)
        rv = -self.std + 2.0 * bernoulli_rv * self.std
        adv_images = images + rv
        adv_images = torch.clamp(adv_images, min=0, max=1)
        return adv_images


class NoiseDataset(object):
    def __init__(
        self,
        data_directory,
        categories,
        model,
        model_name,
        model_type,
        dataset,
        layer,
        logger,
        attack_type,
        probability,
        noise_level,
        batch_size,
        timestamp,
    ):
        self.noise_level = noise_level
        self.model_type = model_type
        self.model = model
        self.probability = probability
        self.output_data_directory = get_noise_attack_data_directory(
            data_directory,
            model_name,
            dataset,
            layer,
            attack_type,
            self.probability,
            self.noise_level,
            timestamp,
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.batch_size = batch_size

        # remove output directory if it exists
        if os.path.exists(self.output_data_directory):
            logger.info(f"Removing {self.output_data_directory}")
            os.system(f"rm -rf {self.output_data_directory}")

        # append image directory to output directory
        self.output_data_directory = f"{self.output_data_directory}/images"

        # create output directory, if it doesn't exist
        if not os.path.exists(self.output_data_directory):
            logger.info(f"Creating {self.output_data_directory}")
            os.makedirs(self.output_data_directory)

        # create dataset
        self.data = SegmentationData(data_directory, categories=categories)
        self.loader = SegmentationPrefetcher(
            self.data, categories=["image"], once=True, batch_size=self.batch_size
        )

        # setup attack
        if attack_type == "gaussian":
            logger.info("Gaussian attack type")
            self.noise_attack = torchattacks.GN(self.model, self.noise_level)
        elif attack_type == "bernoulli":
            logger.info("Bernoulli attack type")
            self.noise_attack = BernoulliNoise(self.model, self.noise_level)
        elif attack_type == "uniform":
            logger.info("Uniform attack type")
            self.noise_attack = UniformNoise(self.model, self.noise_level)
        else:
            raise NotImplementedError

        # create pool for multiprocessing
        self.pool = ThreadPool(processes=16)
        logger.info("Running Noise Attack with probability {}".format(self.probability))

    def create(self):
        loader = self.loader
        num_batches = (len(loader.indexes) + loader.batch_size - 1) / loader.batch_size

        for batch_idx, batch in enumerate(
            loader.tensor_batches(model_type=self.model_type)
        ):

            # log current progress
            self.logger.info(f"Processing batch {batch_idx}/{num_batches}")

            input = torch.from_numpy(batch[0]).to(self.device)  # (batch_size, 3, H, W)
            adv_input = self.noise_attack(input)  # (batch_size, 3, H, W)
            adv_input = adv_input.detach().cpu()

            for i in range(adv_input.shape[0]):
                if np.random.rand() > self.probability:
                    continue
                # Example:
                # file_path:  /home/ddivyansh/Research/NetDissectionLite/dataset/broden1_224/images/opensurfaces/001.jpg
                # output_data_directory:  /home/ddivyansh/Research/NetDissectionLite/dataset/broden1_224_gaussian_noise_0.1/images/
                # parent_dir: opensurfaces
                # file_name:  000001.jpg
                # adversarial_file_directory: /home/ddivyansh/Research/NetDissectionLite/dataset/broden1_224_gaussian_noise_0.1/images/opensurfaces/
                file_path = self.data.filename(i + batch_idx * loader.batch_size)
                parent_directory = os.path.dirname(file_path).split(os.sep)[-1]
                file_name = os.path.basename(file_path)

                # save image to output directory
                # we always have batch_size = 1
                adversarial_file_directory = (
                    f"{self.output_data_directory}/{parent_directory}"
                )
                self.pool.apply_async(
                    NoiseDataset._save,
                    args=(adv_input[i], adversarial_file_directory, file_name),
                )

        # wait for pool to finsh all tasks
        self.pool.close()
        self.pool.join()

    @staticmethod
    def _save(image, adversarial_file_directory, file_name):
        if not os.path.exists(adversarial_file_directory):
            os.makedirs(adversarial_file_directory)
        save_image_from_tensor(
            image,
            f"{adversarial_file_directory}/{file_name}",
            normalized=True,
        )

        # sleep thread until os read succeeds
        while not os.path.exists(f"{adversarial_file_directory}/{file_name}"):
            time.sleep(0.1)
