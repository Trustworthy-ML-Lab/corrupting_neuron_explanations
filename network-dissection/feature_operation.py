import multiprocessing.pool as pool
import os
import time

import numpy as np
import torch
from scipy.misc import imresize

import settings
from loader.data_loader import SegmentationData, SegmentationPrefetcher, load_csv
from models.normalize import Normalize
from util.common import get_normalization_layer
from util.runningstats import Quantile

logger = settings.logger

features_blobs = []


class FeatureOperator:
    def __init__(self, model, units=None):
        self.model = model
        self.data = SegmentationData(
            settings.DATA_DIRECTORY, categories=settings.CATAGORIES
        )

        # get normalization layer
        self.norm = get_normalization_layer(settings.MODEL_TYPE)

        # setup feature shape
        loader = self.get_loader(batch_size=1)
        dummy_batch = next(loader.tensor_batches(model_type=settings.MODEL_TYPE))
        dummy_features = FeatureOperator.get_features(
            self.model, self.norm, dummy_batch[0], units=units
        )
        self.feature_shape = (
            dummy_features.shape[1],
            dummy_features.shape[2],
            dummy_features.shape[3],
        )  # (units, height, width)
        logger.info(f"feature shape: {self.feature_shape}")

    def get_loader(self, batch_size=settings.BATCH_SIZE):
        return SegmentationPrefetcher(
            self.data, categories=["image"], once=True, batch_size=batch_size
        )

    @staticmethod
    def get_features(model, norm, batch, units=None):
        input = batch[0]
        input = torch.from_numpy(input).cuda()

        if (
            units is None
            and settings.CONCEPT_ATTACK
            and settings.CONCEPT_ATTACK_RUN_PARTIAL
        ):
            units = [settings.CONCEPT_ATTACK_UNIT]

        # extract features
        with torch.no_grad():
            input = norm(input)
            features = model.forward(input)

            # units is a list of units to extract or None for all units
            if units is not None:
                features = features[:, units, :, :]

        return features

    def quantile_threshold(self, savepath="", units=None):
        qtpath = os.path.join(settings.OUTPUT_FOLDER, savepath)
        if savepath and os.path.exists(qtpath):
            ret = np.load(qtpath)
            ret = torch.from_numpy(ret).cuda()
            return ret

        logger.info("calculating quantile threshold")
        quant = Quantile(r=300 * 1024, seed=1)
        batch_size = settings.BATCH_SIZE
        loader = SegmentationPrefetcher(
            self.data, categories=["image"], once=True, batch_size=batch_size
        )
        maxfeatures = torch.zeros(
            (len(loader.indexes), self.feature_shape[0]), device="cuda"
        )  # (num_images, units)
        count = 0
        start_time = time.time()
        last_batch_time = start_time
        for batch_idx, batch in enumerate(
            loader.tensor_batches(model_type=settings.MODEL_TYPE)
        ):
            # calculate statistics on batch processing speed
            batch_time = time.time()
            rate = batch_idx / (batch_time - start_time + 1e-15)
            batch_rate = batch_size / (batch_time - last_batch_time + 1e-15)
            last_batch_time = batch_time

            # process batch
            logger.info(
                "Processing batch %d: rate %f , batch_rate %f"
                % (batch_idx, rate, batch_rate)
            )
            batch = FeatureOperator.get_features(
                self.model, self.norm, batch, units=units
            )  # (batch_size, units, height, width)

            # update max features
            maxfeatures[count : count + batch.size(0), :] = torch.max(
                torch.max(batch, dim=3)[0], dim=2
            )[0]
            count += batch.size(0)

            # update quantile
            batch = torch.permute(batch, (0, 2, 3, 1)).reshape(
                -1, self.feature_shape[0]
            )
            quant.add(batch)

        # calculate threshold for each unit
        ret = quant.readout(1000)[:, int(1000 * (1 - settings.QUANTILE) - 1)]
        if savepath:
            np.save(qtpath, ret.detach().cpu().numpy())
        return ret, maxfeatures

    @staticmethod
    def get_bin_count(input):
        """
        Get frequency items in the range [0, val)
        Works only using matrix multiplication
        Supports vectorized operation
        input: long = batch * samples
        output: batch * val
        """
        batch_size = input.shape[0]
        samples = input.shape[1]
        bincount = input.max().item() + 1
        device = input.device

        # setup sparse torch matrix of size (batch_size, bincount, samples)
        x = (
            torch.arange(batch_size, device=device)
            .view(-1, 1)
            .repeat(1, samples)
            .ravel()
        )
        y = input.ravel()
        z = torch.arange(samples, device=device).repeat(batch_size)
        indices = torch.stack([x, y, z], dim=0)
        val = torch.sparse_coo_tensor(
            indices,
            torch.ones(indices.shape[1], device=device, dtype=torch.float64),
            torch.Size([batch_size, bincount, samples]),
        )

        # get bin count
        bincounts = torch.sparse.sum(val, dim=2).to_dense()
        return bincounts

    def tally_job(self, threshold):
        # setup constants
        model = self.model
        feature_shape = self.feature_shape
        norm = self.norm
        data = self.data
        start = 0
        end = self.data.size()

        # number of units
        units = feature_shape[0]

        # number of concepts in the dataset
        # Ex: {'number': 200, 'name': 'palm', 'category': {'object': 405}, 'frequency': 405, 'coverage': 25.589818, 'syns': ['palm tree']}
        labels = len(self.data.label)

        # name of the categories
        # Ex: ['color', 'object', 'part', 'material', 'scene', 'texture']
        categories = self.data.category_names()

        # maps concept to category
        label_cat = (
            torch.from_numpy(data.labelcat).to(torch.float64).cuda()
        )  # (num_concepts, num_categories)

        # intersection of pixels for (unit, concept) pair
        tally_both = torch.zeros((units, labels), dtype=torch.float64).cuda()

        # number of pixels above threshold for a unit across all images in the dataset
        tally_units = torch.zeros(units, dtype=torch.float64).cuda()

        # tally_units for categories which are not empty
        tally_units_cat = torch.zeros(
            (units, len(categories)), dtype=torch.float64
        ).cuda()

        # number of pixels for a given concept across all images in the dataset
        tally_labels = torch.zeros(labels, dtype=torch.float64).cuda()

        # setup data loaders
        pd = SegmentationPrefetcher(
            data,
            categories=data.category_names(),
            once=True,
            batch_size=settings.TALLY_BATCH_SIZE,
            ahead=settings.TALLY_AHEAD,
            start=start,
            end=end,
        )
        loader = SegmentationPrefetcher(
            data,
            categories=["image"],
            once=True,
            batch_size=settings.BATCH_SIZE,
            start=start,
            end=end,
        )

        # calculate tally
        count = start
        start_time = time.time()
        last_batch_time = start_time
        for batch_idx, (batch, input) in enumerate(
            zip(pd.batches(), loader.tensor_batches(model_type=settings.MODEL_TYPE))
        ):
            # calculate statistics on batch processing speed
            batch_time = time.time()
            rate = (count - start) / (batch_time - start_time + 1e-15)
            batch_rate = len(batch) / (batch_time - last_batch_time + 1e-15)
            last_batch_time = batch_time

            # get features and upsample
            features = FeatureOperator.get_features(model, norm, input)
            features = torch.nn.functional.interpolate(
                features, size=(batch[0]["sh"], batch[0]["sw"]), mode="bilinear"
            )

            # calculate tally
            logger.info(
                "labelprobe image index %d, items per sec %.4f, %.4f"
                % (batch_idx, rate, batch_rate)
            )
            for concept_map_idx, concept_map in enumerate(batch):
                count += 1
                pixels = []
                feature_map = features[concept_map_idx]  # (units, H, W)
                mask = feature_map > threshold.reshape(
                    (-1, 1, 1)
                )  # threshold is (units, 1, 1)

                # concatenate all concept maps
                pixels = np.concatenate(
                    [concept_map[cat] for cat in data.category_names()]
                )
                pixels = torch.from_numpy(pixels).cuda()  # (concepts, H, W)

                # increment count of pixels for each concept
                # count frequence of each concept in range (0, labels)
                tally_label = FeatureOperator.get_bin_count(
                    pixels.ravel().unsqueeze(0)
                )[
                    0
                ]  # (num_concepts)
                tally_label[0] = 0
                tally_labels[: tally_label.shape[0]] = torch.add(
                    tally_labels[: tally_label.shape[0]], tally_label
                )

                # calculate tally of units
                tally_unit = torch.sum(
                    mask, dim=(1, 2), dtype=torch.float64
                )  # (units,)
                tally_units = torch.add(tally_units, tally_unit)

                # tally_both stores intersection of features and concepts
                intersect = (
                    pixels[None, :, :, :].repeat((units, 1, 1, 1)) * mask[:, None, :, :]
                )  # (unit, concepts sh, sw)
                intersect = intersect.reshape((units, -1))  # (units, concepts * H * W)
                tally_bt = FeatureOperator.get_bin_count(
                    intersect
                )  # TODO: this is slow, can we do this faster? (units, num_concepts)
                tally_bt[:, 0] = 0
                tally_both[:, : tally_bt.shape[1]] = torch.add(
                    tally_both[:, : tally_bt.shape[1]], tally_bt
                )

                # tally_units_cat stores count of units in each category
                tally_cat = torch.matmul(
                    tally_bt, label_cat[: tally_bt.shape[1], :]
                )  # (units, num_categories)
                tally_units_cat = torch.add(
                    tally_units_cat, tally_unit.unsqueeze(1) * (tally_cat > 0)
                )

        # move to cpu
        tally_both = tally_both.cpu().numpy()
        tally_units = tally_units.cpu().numpy()
        tally_units_cat = tally_units_cat.cpu().numpy()
        tally_labels = tally_labels.cpu().numpy()

        return tally_both, tally_units, tally_units_cat, tally_labels

    def tally(self, threshold, savepath=""):
        csvpath = os.path.join(settings.OUTPUT_FOLDER, savepath)
        units = self.feature_shape[
            0
        ]  # shape[0] is the number of units in the convolutional layer

        # name of the categories
        # Ex: ['color', 'object', 'part', 'material', 'scene', 'texture']
        categories = self.data.category_names()

        # get tally for unit and concept
        tally_both, _, tally_units_cat, tally_labels = self.tally_job(threshold)

        # calculate IOU for each (unit, concept) pair
        primary_categories = self.data.primary_categories_per_index()
        tally_units_cat = np.dot(tally_units_cat, self.data.labelcat.T)
        iou = tally_both / (
            tally_units_cat + tally_labels[np.newaxis, :] - tally_both + 1e-10
        )  # (units, concepts)
        pciou = np.array(
            [
                iou * (primary_categories[np.arange(iou.shape[1])] == ci)[np.newaxis, :]
                for ci in range(len(self.data.category_names()))
            ]
        )
        label_pciou = pciou.argmax(axis=2)  # (category, units, concepts)

        # find top 5 highest IOU for each unit
        top_5_iou = np.zeros((units, 5), dtype=np.float64)
        top_5_labels = []
        top_5_categories = []
        for unit_id in range(units):
            top_5_indices = np.argsort(iou[unit_id, :])[::-1][:5]
            top_5_categories_index = [primary_categories[l] for l in top_5_indices]
            top_5_labels.append([self.data.label[l] for l in top_5_indices])
            top_5_categories.append(
                [self.data.category_names()[ci] for ci in top_5_categories_index]
            )
            top_5_iou[unit_id, :] = iou[unit_id, top_5_indices]

        name_pciou = [
            [self.data.name(None, j) for j in label_pciou[ci]]
            for ci in range(len(label_pciou))
        ]
        score_pciou = pciou[
            np.arange(pciou.shape[0])[:, np.newaxis],
            np.arange(pciou.shape[1])[np.newaxis, :],
            label_pciou,
        ]
        bestcat_pciou = score_pciou.argsort(axis=0)[::-1]
        ordering = score_pciou.max(axis=0).argsort()[::-1]
        rets = [None] * len(ordering)

        for i, unit in enumerate(ordering):
            # Top images are top[unit]
            bestcat = bestcat_pciou[0, unit]
            data = {
                "unit": (unit + 1),
                "category": categories[bestcat],
                "label": name_pciou[bestcat][unit],
                "score": score_pciou[bestcat][unit],
            }
            for ci, cat in enumerate(categories):
                label = label_pciou[ci][unit]
                data.update(
                    {
                        "%s-label" % cat: name_pciou[ci][unit],
                        "%s-truth" % cat: tally_labels[label],
                        "%s-activation" % cat: tally_units_cat[unit, label],
                        "%s-intersect" % cat: tally_both[unit, label],
                        "%s-iou" % cat: score_pciou[ci][unit],
                    }
                )

            # write top-5 unit info data
            for k_iter in range(5):
                data.update(
                    {
                        "top%d-label" % (k_iter + 1): top_5_labels[unit][k_iter],
                        "top%d-category" % (k_iter + 1): top_5_categories[unit][k_iter],
                        "top%d-iou" % (k_iter + 1): top_5_iou[unit, k_iter],
                    }
                )

            rets[i] = data

        if savepath:
            import csv

            csv_fields = sum(
                [
                    [
                        "%s-label" % cat,
                        "%s-truth" % cat,
                        "%s-activation" % cat,
                        "%s-intersect" % cat,
                        "%s-iou" % cat,
                    ]
                    for cat in categories
                ],
                ["unit", "category", "label", "score"],
            )

            # append top 5 level fields to csv
            csv_fields += sum(
                [
                    [
                        "top%d-label" % (k + 1),
                        "top%d-category" % (k + 1),
                        "top%d-iou" % (k + 1),
                    ]
                    for k in range(5)
                ],
                [],
            )

            with open(csvpath, "w") as f:
                writer = csv.DictWriter(f, csv_fields)
                writer.writeheader()
                for i in range(len(ordering)):
                    writer.writerow(rets[i])
        return rets
