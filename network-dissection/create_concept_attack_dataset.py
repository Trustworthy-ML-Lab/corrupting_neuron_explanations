import logging
import os
from multiprocessing.pool import ThreadPool

import numpy as np

from concept_adversary import FGSMConceptAdversary, PGDConceptAdversary
from loader.data_loader import (
    SegmentationData,
    SegmentationPrefetcher,
    get_concept_attack_data_directory,
)
from util.common import get_category_and_index, save_image_from_tensor


class ConceptAdversaryDataset(object):
    def __init__(
        self,
        data_directory,
        categories,
        model,
        model_name,
        model_type,
        dataset,
        layer,
        unit,
        loss_type,
        method,
        epsilon,
        steps,
        target_category,
        target_index,
        target_name,
        batch_size,
        timestamp,
    ):
        self.model_type = model_type
        self.target_category = target_category
        self.target_index = target_index

        # create logger
        self.logger = logging.getLogger(__name__)

        # setup output directory
        self.output_data_directory = get_concept_attack_data_directory(
            data_directory,
            model_name,
            dataset,
            layer,
            unit,
            None,
            method,
            epsilon,
            steps,
            self.target_category,
            target_name,
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
        self.segmentation_loader = SegmentationPrefetcher(
            self.data,
            categories=self.data.category_names(),
            once=True,
            batch_size=batch_size,
        )

        # setup concept adversary
        if method == "pgd":
            self.concept_adversary = PGDConceptAdversary(
                model,
                model_name,
                model_type,
                layer,
                unit,
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
                unit,
                loss_type,
                epsilon=epsilon,
            )
        else:
            raise NotImplementedError

        # setup source and target category
        (self.source_category, self.source_index,) = get_category_and_index(
            self.data, layer=layer, unit=unit, model=model_name, dataset=dataset
        )

        # create pool for multiprocessing
        self.pool = ThreadPool(processes=16)

        # log hyperparameters
        self.logger.info(
            f"Hyperparameters: Layer: {layer}, Unit: {unit}, Source Category: {self.source_category}, Source Index: {self.source_index}, Target Category: {self.target_category}, Target Index: {self.target_index}"
        )

    def create(self):
        loader = self.loader
        num_batches = (len(loader.indexes) + loader.batch_size - 1) / loader.batch_size

        for idx, batch in enumerate(
            zip(
                loader.tensor_batches(model_type=self.model_type),
                self.segmentation_loader.batches(),
            )
        ):
            # batch is a tuple of (image, segmentation)
            # batch[0][0] contains the image tensor. batch[0][0] has shape (batch_size, 3, height, width)
            # batch[1] contains the segmentation info. batch[1][i] is a dictionary segmentation_info

            # log current progress
            self.logger.info(f"Processing batch {idx}/{num_batches}")

            # setup input image tensor
            input = batch[0][0]  # (batch_size, 3, H, W)
            batch_size = input.shape[0]

            # setup segmentation tensor
            segmentation = batch[1]  # (batch_size)
            segmentation_width = segmentation[0]["sw"]
            segmentation_height = segmentation[0]["sh"]
            source_map = np.zeros(
                (input.shape[0], 1, segmentation_width, segmentation_height)
            )  # (batch_size, 1, H, W)
            target_map = np.zeros(
                (input.shape[0], 1, segmentation_width, segmentation_height)
            )  # (batch_size, 1, H, W)
            attack_idx = np.zeros(batch_size, dtype=np.bool)

            # create source and target maps
            for i in range(batch_size):
                # setup source and target ground truth
                if segmentation[i][self.source_category].shape[0] != 0:
                    # segmentation[i][self.source_category] is np.darray of size (1, H, W)
                    # combine all the boolean maps into one of shape (1, H, W)
                    concept_map_source = (
                        segmentation[i][self.source_category] == self.source_index
                    )  # (num_instances, H, W)
                    concept_map_source = np.sum(concept_map_source, axis=0) > 0
                    source_map[i] = concept_map_source * 1  # (1, H, W)

                if (
                    self.target_category
                    and segmentation[i][self.target_category].shape[0] != 0
                ):
                    # segmentation[i][self.target_category] is np.darray of size (1, H, W)
                    concept_map_target = (
                        segmentation[i][self.target_category] == self.target_index
                    )
                    concept_map_target = np.sum(concept_map_target, axis=0) > 0
                    target_map[i] = concept_map_target * 1  # (1, H, W)

                # if either source map or target map have no overlap, skip
                if (source_map[i].sum() != 0) or (target_map[i].sum() != 0):
                    attack_idx[i] = True

            # get adversarial image if any attack indices are True
            if np.any(attack_idx):
                adv_input = self.concept_adversary.attack(
                    input[attack_idx],
                    source_map[attack_idx],
                    target_map[attack_idx],
                    iw=segmentation_width,
                    ih=segmentation_height,
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
                assert self.data.filename(id) == batch[1][i]["fn"]
                file_path = self.data.filename(id)
                parent_directory = os.path.dirname(file_path).split(os.sep)[-1]
                file_name = os.path.basename(file_path)

                # save image to output directory
                # we always have batch_size = 1
                adversarial_file_directory = (
                    f"{self.output_data_directory}/{parent_directory}"
                )
                self.pool.apply_async(
                    ConceptAdversaryDataset._save,
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
