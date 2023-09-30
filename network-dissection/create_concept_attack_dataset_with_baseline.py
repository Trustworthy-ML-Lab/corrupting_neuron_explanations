import logging
import os
from multiprocessing.pool import ThreadPool

import numpy as np
import torch

from concept_adversary import FGSMConceptAdversary, PGDConceptAdversary
from loader.data_loader import (
    SegmentationData,
    SegmentationPrefetcher,
    get_concept_attack_data_directory,
)
from util.common import save_image_from_tensor


class ConceptAdversaryDatasetBaseline(object):
    def __init__(
        self,
        data_directory,
        categories,
        model,
        model_name,
        model_type,
        dataset,
        layer,
        source_unit,
        target_unit,
        source_threshold,
        target_threshold,
        loss_type,
        method,
        epsilon,
        steps,
        batch_size,
        timestamp,
    ):
        self.model = model
        self.model_type = model_type
        self.source_unit = source_unit
        self.target_unit = target_unit
        self.source_threshold = source_threshold
        self.target_threshold = target_threshold

        # create logger
        self.logger = logging.getLogger(__name__)

        # setup output directory
        self.output_data_directory = get_concept_attack_data_directory(
            data_directory,
            model_name,
            dataset,
            layer,
            source_unit,
            target_unit,
            method,
            epsilon,
            steps,
            None,
            None,
            timestamp,
        )

        # remove output directory if it exists
        if os.path.exists(self.output_data_directory):
            self.logger.info(f"Removing {self.output_data_directory}")
            os.system(f"rm -rf {self.output_data_directory}")

        # append image directory to output directory
        self.output_data_directory = f"{self.output_data_directory}/images"

        # create output directory, if it doesn't exist
        if not os.path.exists(self.output_data_directory):
            self.logger.info(f"Creating {self.output_data_directory}")
            os.makedirs(self.output_data_directory)

        # setup image and sgementation loader
        self.data = SegmentationData(data_directory, categories=categories)
        self.loader = SegmentationPrefetcher(
            self.data, categories=["image"], once=True, batch_size=batch_size
        )

        # setup concept adversary
        if method == "pgd":
            self.concept_adversary = PGDConceptAdversary(
                model,
                model_name,
                model_type,
                layer,
                source_unit,
                loss_type,
                epsilon=epsilon,
                steps=steps,
            )
        elif method == "fgsm":
            self.concept_adversary = FGSMConceptAdversary(
                model,
                model_name,
                model_type,
                layer,
                source_unit,
                loss_type,
                epsilon=epsilon,
            )
        else:
            raise NotImplementedError

        # create pool for multiprocessing
        self.pool = ThreadPool(processes=16)

        # log hyperparameters
        self.logger.info(
            f"Hyperparameters: Layer: {layer}, Source Unit: {source_unit}, Target Unit: {target_unit}, Source Threshold: {source_threshold}, Target Threshold: {target_threshold}, Loss Type: {loss_type}, Method: {method}, Epsilon: {epsilon}, Steps: {steps}, Batch Size: {batch_size}"
        )

    def create(self):
        self.logger.info("Running without baseline")
        loader = self.loader
        num_batches = (len(loader.indexes) + loader.batch_size - 1) / loader.batch_size

        for idx, batch in enumerate(loader.tensor_batches(model_type=self.model_type)):
            # batch[0] contains the image tensor. batch[0] has shape (batch_size, 3, height, width)

            # log current progress
            self.logger.info(f"Processing batch {idx}/{num_batches}")

            # setup input image tensor
            input = batch[0]  # (batch_size, 3, H, W)
            input = torch.from_numpy(input).cuda()
            batch_size = input.shape[0]

            out = self.model(input)
            source_map = (
                out[:, [self.source_unit]] > self.source_threshold
            )  # (batch_size, 1, H, W)
            target_map = (
                out[:, [self.target_unit]] > self.target_threshold
            )  # (batch_size, 1, H, W)

            # check the number of idx where source_map and target_map
            # have atleast one True value
            attack_idx = (
                torch.sum(source_map, dim=(1, 2, 3))
                + torch.sum(target_map, dim=(1, 2, 3))
            ) > 0

            # get adversarial image if any attack indices are True
            if torch.any(attack_idx):
                adv_input = self.concept_adversary.attack(
                    input[attack_idx],
                    source_map[attack_idx],
                    target_map[attack_idx],
                    iw=source_map.shape[2],
                    ih=source_map.shape[3],
                )

            # save images
            adv_input_idx = 0
            for i in range(batch_size):
                # skip if no attack
                if not attack_idx[i]:
                    continue

                # Example:
                # file_path:  /home/ddivyansh/Research/NetDissectionLite/dataset/broden1_224/images/opensurfaces/001.jpg
                # output_data_directory:  /home/ddivyansh/Research/NetDissectionLite/dataset/broden1_224_gaussian_noise_0.1/images/
                # parent_dir: opensurfaces
                # file_name:  000001.jpg
                # adversarial_file_directory: /home/ddivyansh/Research/NetDissectionLite/dataset/broden1_224_gaussian_noise_0.1/images/opensurfaces/
                id = i + idx * loader.batch_size
                file_path = self.data.filename(id)
                parent_directory = os.path.dirname(file_path).split(os.sep)[-1]
                file_name = os.path.basename(file_path)

                # save image to output directory
                # we always have batch_size = 1
                adversarial_file_directory = (
                    f"{self.output_data_directory}/{parent_directory}"
                )
                self.pool.apply_async(
                    ConceptAdversaryDatasetBaseline._save,
                    args=(
                        adv_input[adv_input_idx].detach().cpu(),
                        adversarial_file_directory,
                        file_name,
                    ),
                )
                adv_input_idx += 1

        # close pool
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
