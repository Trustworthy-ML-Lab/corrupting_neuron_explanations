import csv

import numpy as np
from PIL import Image

from models.normalize import Normalize


def santize_path_name(path: str):
    """
    Sanitizes a path name by replacing invalid characters with underscores.
    """
    return path.replace(" ", "_").replace(".", "_")


def get_noise_attack_data_directory(
    path: str,
    model: str,
    dataset: str,
    layer: str,
    noise_type: str,
    probability: float,
    std: float,
    timestamp: str,
):
    output_path = f"_modified/{model}/{dataset}/{layer}/noise/{noise_type}/poison_prob_{probability}/noise_{std}/{timestamp}"
    output_path = santize_path_name(output_path)
    return path + output_path


def get_concept_attack_data_directory(
    path: str,
    model: str,
    dataset: str,
    layer: str,
    source_unit: str,
    target_unit: str,
    method: str,
    epsilon: float,
    steps: int,
    target_category: str,
    target_name: int,
    timestamp: str,
):
    output_path = f"_modified/{model}/{dataset}/{layer}/concept_attack/source_unit_{source_unit}/method_{method}/epsilon_{epsilon}/steps_{steps}"

    if target_unit is not None:
        output_path += f"/target_unit_{target_unit}"
    elif target_category is not None:
        output_path += f"/target_{target_category}/target_name_{target_name}"
    else:
        output_path += "/untargeted"

    output_path += f"/{timestamp}"
    output_path = santize_path_name(output_path)
    return path + output_path


def image_to_tensor(path, size=None, normalized=False):
    """
    Converts a PIL image to a float32 numpy array
    """
    image = Image.open(path)

    # resize image, if specified
    if size:
        image = image.resize((size, size))

    # convert to numpy array
    image = np.array(image)

    # normalize to the range [0, 1]
    if normalized:
        image = image.astype(np.float32)
        image = image / 255.0

    return image


def tensor_to_image(tensor, normalized=False):
    """
    Converts a tensor to a PIL image.
    """
    if normalized:
        tensor = tensor * 255.0

    # convert tensor to channel first
    tensor = tensor.numpy().astype(np.uint8)
    tensor = np.transpose(tensor, (1, 2, 0))
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def save_image_from_tensor(tensor, filename, normalized=False):
    """
    Converts a tensor to a PIL image and saves it to a file.
    """
    image = tensor_to_image(tensor, normalized)
    image.save(filename)


def instrumented_layername(model, layer):
    """Chooses the layer name to dissect."""
    if layer is not None:
        if model == "vgg16":
            return "features." + layer
        elif model.startswith("resnet"):
            return "module." + layer
        return layer

    # Default layers to probe
    if model == "alexnet":
        return "conv5"
    elif model == "vgg16":
        return "features.conv5_3"
    elif model == "resnet152":
        return "7"
    elif model == "progan":
        return "layer4"
    elif model == "resnet50":
        return "module.layer4"


def get_normalization_layer(model_type):
    # setup normalization layer
    if model_type == "pytorch":
        # follows RGB order
        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif model_type == "caffe":
        # follows BGR order
        norm = Normalize(
            mean=255.0 * np.array([0.406, 0.456, 0.485]),
            std=255.0 * np.array([0.225, 0.224, 0.229]),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return norm


def get_category_and_index(data, layer, unit, model="vgg16", dataset="places365"):
    """
    Get category and index from layer and unit
    """
    # open csv file
    # csv file has format:
    # unit,category,label,score,color-label,color-truth,color-activation,color-intersect,color-iou,object-label,object-truth,object-activation,object-intersect,object-iou,part-label,part-truth,part-activation,part-intersect,part-iou,material-label,material-truth,material-activation,material-intersect,material-iou,scene-label,scene-truth,scene-activation,scene-intersect,scene-iou,texture-label,texture-truth,texture-activation,texture-intersect,texture-iou,top1-label,top1-category,top1-iou,top2-label,top2-category,top2-iou,top3-label,top3-category,top3-iou,top4-label,top4-category,top4-iou,top5-label,top5-category,top5-iou
    with open(f"./tally/{model}/{dataset}/{layer}/tally.csv", "r") as csv_file:
        # get category and label for unit
        reader = csv.DictReader(csv_file)
        for row in reader:
            # we follow 0 based indexing
            # tally follows 1 based indexing
            if not int(row["unit"]) == (unit + 1):
                continue
            label_info = [
                entry
                for entry in data.category_label[row["category"]]
                if entry["name"] == row["label"]
            ][0]
            return row["category"], label_info["number"]


def get_tally_label(layer, unit, model="vgg16", quantile=None):
    """
    Get category and index from layer and unit
    """
    # open csv file
    # csv file has format:
    # unit,category,label,score,color-label,color-truth,color-activation,color-intersect,color-iou,object-label,object-truth,object-activation,object-intersect,object-iou,part-label,part-truth,part-activation,part-intersect,part-iou,material-label,material-truth,material-activation,material-intersect,material-iou,scene-label,scene-truth,scene-activation,scene-intersect,scene-iou,texture-label,texture-truth,texture-activation,texture-intersect,texture-iou,top1-label,top1-category,top1-iou,top2-label,top2-category,top2-iou,top3-label,top3-category,top3-iou,top4-label,top4-category,top4-iou,top5-label,top5-category,top5-iou

    filepath = f"./tally/{model}/{layer}/tally.csv"
    if quantile is not None:
        filepath = f"tally/{model}/quantile_{quantile}/{layer}"
        filepath = filepath.replace(".", "_")
        filepath = f"./{filepath}/tally.csv"
    with open(filepath, "r") as csv_file:
        # get category and label for unit
        reader = csv.DictReader(csv_file)
        for row in reader:
            # we follow 0 based indexing
            # tally follows 1 based indexing
            if not int(row["unit"]) == (unit + 1):
                continue
            return row["label"]

    raise ValueError(f"Could not find label for layer {layer} and unit {unit}")


def label_index_to_name(category, index, dataset_path="./dataset/broden1_224"):
    """
    Convert label index to label name
    """
    # open c_<category>.txt
    with open(f"{dataset_path}/c_{category}.csv", "r") as csv_file:
        # get label name for index
        reader = csv.DictReader(csv_file)
        for row in reader:
            if not int(row["number"]) == index:
                continue
            return row["name"]


def label_name_to_index(category, name, dataset_path="./dataset/broden1_224"):
    """
    Convert label name to label index
    """
    # open c_<category>.txt
    with open(f"{dataset_path}/c_{category}.csv", "r") as csv_file:
        # get label name for index
        reader = csv.DictReader(csv_file)
        for row in reader:
            if not row["name"] == name:
                continue
            return int(row["number"])

    raise ValueError(f"Could not find index for category {category} and name {name}")


def get_threshold(model, layer):
    """
    Get threshold for baseline model for a specific layer
    """
    threshold = np.load(f"quantile/{model}/{layer}/quantile.npy")
    return threshold
