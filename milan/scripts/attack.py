"""Dissect a pretrained vision model."""
import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch import cuda
from tqdm import tqdm

from src import milan, milannotations
from src.attack.attack_dataset import AttackDataset
from src.attack.fgsm import FGSM
from src.attack.noise_dataset import NoiseDataset
from src.attack.pgd import PGD
from src.deps.netdissect.tally import make_loader
from src.exemplars import compute, datasets, models, transforms
from src.loss.dissection_loss import DissectionLoss
from src.utils import env, logging
from src.utils.common import (
    MODEL_TANSFORMS,
    remove_transforms_for_adversarial_attack,
    shrink_to_layer,
)

parser = argparse.ArgumentParser(description="compute unit exemplars")

# setup base arguments
parser.add_argument("model", type=str, help="model architecture")
parser.add_argument("dataset", type=str, help="dataset model is trained on")
parser.add_argument("--layer", type=str, help="layer names to compute exemplars for")
parser.add_argument(
    "--units", type=int, help="only compute exemplars for a given unit (default: all)"
)
parser.add_argument(
    "--experiment", type=str, help="experiment name", default="baseline"
)
parser.add_argument(
    "--batch-size", type=int, default=128, help="batch size (default: 128)"
)
parser.add_argument(
    "--num-workers", type=int, default=16, help="number of worker threads (default: 16)"
)
parser.add_argument(
    "--k", type=int, default=15, help="number of exemplars to compute (default: 15)"
)
parser.add_argument(
    "--quantile-r",
    type=int,
    default=4096 * 64,
    help="parameter r for running quantile (default: 4096*64)",
)
parser.add_argument("--d", action="store_true", help="debug mode (default: False)")
parser.add_argument(
    "--device", help="manually set device (default: guessed)", default="cuda"
)
parser.add_argument(
    "--viz", help="visualize exemplars (default: False)", action="store_true"
)

# setup decoder arguments
parser.add_argument("--no-predict", help="do not predict labels", action="store_true")
parser.add_argument(
    "--milan",
    default=milannotations.KEYS.BASE,
    help="milan model to use (default: base)",
)
parser.add_argument(
    "--temperature", type=float, default=0.2, help="pmi temperature (default: .2)"
)
parser.add_argument(
    "--beam-size", type=int, default=50, help="beam size to rerank (default: 50)"
)

# setup noise attack arguments
parser.add_argument("--noise", action="store_true", help="attack with noise")
parser.add_argument("--noise-type", type=str, default="gaussian")
parser.add_argument(
    "--noise-std",
    type=float,
    default=0.01,
    help="standard deviation of noise to add (default: 0.01)",
)
parser.add_argument(
    "--attack-probability",
    type=float,
    default=1.0,
    help="probability of an image being attacked (default: 1.0)",
)

# setup attack arguments
parser.add_argument("--attack", action="store_true", help="attack the model")
parser.add_argument("--attack-method", default="fgsm", help="attack method")
parser.add_argument("--eps", type=float, default=0.01, help="Attack eps")
parser.add_argument("--pgd-steps", type=int, default=10, help="PGD steps")
parser.add_argument(
    "--attack-target-layer", type=str, default=None, help="Target layer to attack"
)
parser.add_argument(
    "--attack-target-unit", type=int, default=None, help="Target unit to attack"
)
parser.add_argument(
    "--attack-image-count",
    type=int,
    help="Number of images to attack in the examplar set",
)

parser.add_argument(
    "--accuracy", help="record accuracy on attacked data", action="store_true"
)
parser.set_defaults(accuracy=False)


# parse arguments
args = parser.parse_args()

# setup logging
logger = logging.get_logger("attack_main")
if args.d:
    logging.set_log_level("DEBUG")

# setup base variables
experiment = args.experiment
key = f"{args.model}/{args.dataset}"
layer = args.layer
units = None
units_str = "all"
if args.units is not None:
    units = [args.units]
    units_str = str(args.units)
device = args.device or "cuda" if cuda.is_available() else "cpu"

# setup results root
results_root = env.results_dir() / experiment
results_dir = results_root / args.model / args.dataset / args.layer / units_str

# update folder root for attacks
if args.attack:
    results_dir = (
        results_dir
        / str(args.eps).replace(".", "_")
        / args.attack_method
        / str(args.attack_image_count)
    )

    if args.attack_target_layer is not None:
        results_dir = (
            results_dir / args.attack_target_layer / str(args.attack_target_unit)
        )
elif args.noise:
    results_dir = (
        results_dir
        / args.noise_type
        / str(args.noise_std).replace(".", "_")
        / str(args.attack_probability).replace(".", "_")
    )

# add timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = results_dir / timestamp

# setup visualization root
viz_dir = results_dir / "viz"

# setup logging file
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
logging.set_log_file_handler(results_dir / "output.log")

# dump args
logger.info(f"args: {args}")
logger.info(f"timestamp: {timestamp}")

# Setup model
original_model, layers, config = models.load(
    f"{args.model}/{args.dataset}", map_location=device
)
assert layer in layers, f"Layer {layer} not in {layers}"
model = shrink_to_layer(original_model, layer)
logger.debug(f"Model: {original_model}, shrunk model: {model}")

# Setup dataset
dataset = datasets.load(args.dataset)

# Setup attack
transform_dataloader = None
if args.attack:
    # assert that units is of length 1
    assert len(units) == 1, "Only one unit can be attacked at a time"

    # remove transforms before
    dataset, transform_after_attack = remove_transforms_for_adversarial_attack(
        dataset, logger
    )

    # setup attack dataset
    dataset = AttackDataset(
        args.model,
        args.dataset,
        dataset,
        Path(f"./results"),
        source_layer=layer,
        source_unit=units[0],
        attack_image_count=args.attack_image_count,
        target_layer=args.attack_target_layer,
        target_unit=args.attack_target_unit,
        transform_after_attack=transform_after_attack,
    )

    # setup attack method
    if args.attack_method == "fgsm":
        transform_dataloader = FGSM(
            args.eps, DissectionLoss(), model, transform_after_attack, layer, units[0]
        )
    elif args.attack_method == "pgd":
        transform_dataloader = PGD(
            args.eps,
            args.pgd_steps,
            DissectionLoss(),
            model,
            transform_after_attack,
            layer,
            units[0],
        )
    else:
        raise ValueError(f"Unknown attack method {args.attack_method}")
elif args.noise:
    # remove transforms before
    dataset, transform_after_attack = remove_transforms_for_adversarial_attack(
        dataset, logger
    )

    # setup Noise
    dataset = NoiseDataset(
        dataset,
        args.noise_type,
        args.noise_std,
        args.attack_probability,
        transform_after_attack,
    )

if args.accuracy:
    accuracies = []
    all_labels = []
    loader = make_loader(
        dataset,
        sample_size=None,
        batch_size=args.batch_size,
        device=args.device,
        transform_dataloader=transform_dataloader,
    )
    original_model = original_model.to(device)
    for images, labels in tqdm(loader):
        with torch.no_grad():
            output = original_model(images)
            pred = torch.argmax(output, dim=1)
            accuracies.append(pred == labels)
            all_labels.append(labels)
    accuracies = torch.cat(accuracies, dim=0).cpu()
    all_labels = torch.cat(all_labels, dim=0).cpu()

    accuracy_by_class = {"Class": [], "Accuracy": []}

    accuracy_by_class["Class"].append("Total")
    accuracy_by_class["Accuracy"].append(float(torch.mean(accuracies.float())))

    for i in range(torch.max(all_labels) + 1):
        gt_i = all_labels == i
        tp = gt_i * accuracies
        accuracy_by_class["Class"].append(i)
        accuracy_by_class["Accuracy"].append(
            float(torch.sum(tp.float()) / torch.sum(gt_i.float()))
        )

    df = pd.DataFrame(accuracy_by_class)
    df.to_csv(results_dir / "accuracies.csv", index=False)

# Compute exemplars
compute.run(
    model,
    dataset,
    units=units,
    results_dir=results_dir,
    viz_dir=viz_dir,
    device=device,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    transform_dataloader=transform_dataloader,
    k=args.k,
    quantile_r=args.quantile_r,
    save_viz=args.viz,
    **config.exemplars.kwargs,
)


if not args.no_predict:
    # Load MILAN
    decoder = milan.pretrained(args.milan)
    decoder.to(device)

    # Load top images
    # try to load on GPU first, if that fails, load on CPU
    try:
        milannotations_dataset = milannotations.load(
            key, path=results_dir, device=device
        )
    except RuntimeError:
        milannotations_dataset = milannotations.load(
            key, path=results_dir, device="cpu"
        )

    # Predict!
    predictions = decoder.predict(
        milannotations_dataset,
        strategy="rerank",
        temperature=args.temperature,
        beam_size=args.beam_size,
        device=device,
    )

    # Save predictions
    rows = []
    for index, description in enumerate(predictions):
        sample = milannotations_dataset[index]
        row = (str(sample.layer), str(sample.unit), description)
        rows.append(row)
    results_csv_file = results_dir / "descriptions.csv"
    with results_csv_file.open("w") as handle:
        csv.writer(handle).writerows(rows)
