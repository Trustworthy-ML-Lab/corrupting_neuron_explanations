######### global settings  #########
import argparse
import logging
import os
from datetime import datetime

from util.common import (
    get_concept_attack_data_directory,
    get_noise_attack_data_directory,
    label_name_to_index,
)

# setup argument parser
parser = argparse.ArgumentParser(description="Hyperparameters")

# set up the arguments for network dissection
parser.add_argument("--model", type=str, default="vgg16", help="model name")
parser.add_argument("--layer", type=str, default="conv5_3", help="layer name")
parser.add_argument("--dataset", type=str, default="places365", help="dataset name")

# set up the argument parser for noise attack
parser.add_argument(
    "--noise_attack", action="store_true", default=False, help="use noise attack"
)
parser.add_argument(
    "--noise_attack_type", type=str, default="gaussian", help="noise attack type"
)
parser.add_argument(
    "--noise_attack_probability", type=float, default=1.0, help="noise probability"
)
parser.add_argument("--noise_attack_std", type=float, default=0.03, help="noise std")

# set up the arguments for adversarial attack
parser.add_argument(
    "--concept_attack",
    action="store_true",
    default=False,
    help="whether to use concept attack",
)
parser.add_argument(
    "--concept_attack_type", type=str, default="pgd", help="type of concept attack"
)
parser.add_argument(
    "--concept_attack_unit", type=int, default=190, help="unit in the layer to attack"
)
parser.add_argument(
    "--concept_attack_loss_type",
    type=str,
    default="basic",
    help="loss type for concept attack",
)
parser.add_argument(
    "--concept_attack_run_partial",
    action="store_true",
    default=False,
    help="whether to run partial network dissect only on concept attack unit",
)
parser.add_argument(
    "--concept_attack_pgd_eps",
    type=float,
    default=(4.0 / 255.0),
    help="eps of concept attack",
)
parser.add_argument(
    "--concept_attack_pgd_steps", type=int, default=10, help="iter of concept attack"
)
parser.add_argument(
    "--concept_attack_target_category",
    type=str,
    default=None,
    help="target category",
)
parser.add_argument(
    "--concept_attack_target_name", type=str, default=None, help="target name"
)
parser.add_argument(
    "--concept_attack_target_unit",
    type=int,
    default=None,
    help="target unit in the layer to attack",
)

# setup the arguments for saving the results
parser.add_argument(
    "--base_dir",
    type=str,
    default=os.environ["BASE_DIR"] if os.environ.get("BASE_DIR") else ".",
    help="base directory",
)
parser.add_argument(
    "--experiment_name", type=str, default="speedup", help="experiment name"
)
parser.add_argument(
    "--timestamp",
    type=str,
    default=datetime.now().strftime("%Y%m%d_%H%M%S"),
    help="time stamp",
)
parser.add_argument("--test", action="store_true", default=False, help="test")
parser.add_argument("--quantile", type=float, default=0.005, help="quantile")

# get parsed arguments
args = parser.parse_args()

# set up arguments related to the network dissect
BASE_DIRECTORY = args.base_dir  # base directory
TIMESTAMP = args.timestamp  # time stamp
TEST_MODE = (
    args.test
)  # turning on the testmode means the code will run on a small dataset.
MODEL = args.model
MODEL_TYPE = (  # caffe or pytorch, caffe follow BGR order while pytorch follow RGB order
    "pytorch"
)
DATASET = args.dataset  # model trained on: places365 or imagenet
QUANTILE = args.quantile  # the threshold used for activation
SEG_THRESHOLD = 0.04  # the threshold used for visualization
SCORE_THRESHOLD = 0.04  # the threshold used for IoU score (in HTML file)
TOPN = 10  # to show top N image with highest activation for each unit
PARALLEL = (
    1  # how many process is used for tallying (Experiments show that 1 is the fastest)
)
CATAGORIES = [
    "object",
    "part",
    "scene",
    "texture",
    "color",
    "material",
]  # concept categories that are chosen to detect: "object", "part", "scene", "material", "texture", "color"
LAYER = args.layer  # feature names that are chosen to detect

# set up arguments related to the noise attack
NOISE_ATTACK = args.noise_attack
NOISE_ATTACK_TYPE = args.noise_attack_type
NOISE_ATTACK_PROBABILITY = round(args.noise_attack_probability, 6)
NOISE_ATTACK_STD = round(args.noise_attack_std, 6)

# set up arguments related to the concept attack
CONCEPT_ATTACK = args.concept_attack
CONCEPT_ATTACK_RUN_PARTIAL = args.concept_attack_run_partial
CONCEPT_ATTACK_METHOD = args.concept_attack_type
CONCEPT_ATTACK_METHOD_PGD_EPSILON = round(args.concept_attack_pgd_eps, 6)
CONCEPT_ATTACK_METHOD_PGD_STEPS = args.concept_attack_pgd_steps
CONCEPT_ATTACK_UNIT = args.concept_attack_unit
CONCEPT_ATTACK_TARGET_UNIT = args.concept_attack_target_unit
CONCEPT_ATTACK_LOSS_TYPE = args.concept_attack_loss_type
CONCEPT_ATTACK_TARGET_CATEGORY = args.concept_attack_target_category

########### sub settings ###########
# In most of the case, you don't have to change them.
# DATA_DIRECTORY: where broaden dataset locates
# IMG_SIZE: image size, alexnet use 227x227
# NUM_CLASSES: how many labels in final prediction
# LAYER: layer where features will be extracted
# MODEL_FILE: the model file to be probed, "None" means the pretrained model in torchvision
# MODEL_PARALLEL: some model is trained in multi-GPU, so there is another way to load them.
# WORKERS: how many workers are fetching images
# BATCH_SIZE: batch size used in feature extraction
# TALLY_BATCH_SIZE: batch size used in tallying
# INDEX_FILE: if you turn on the TEST_MODE, actually you should provide this file on your own

# dataset parameters
if DATASET == "places365":
    NUM_CLASSES = 365
elif DATASET == "imagenet":
    NUM_CLASSES = 1000

# model parameters
if MODEL != "alexnet":
    DATA_DIRECTORY = f"{BASE_DIRECTORY}/dataset/broden1_224"
    IMG_SIZE = 224
else:
    DATA_DIRECTORY = f"{BASE_DIRECTORY}/dataset/broden1_227"
    IMG_SIZE = 227

if MODEL == "resnet18":
    LAYER = "layer4"
    if DATASET == "imagenet":
        MODEL_FILE = None
        MODEL_PARALLEL = False
elif MODEL == "resnet152":
    LAYER = "7"
    if DATASET == "places365":
        MODEL_FILE = None
        MODEL_PARALLEL = False
elif MODEL == "vgg16":
    if DATASET == "places365":
        MODEL_FILE = None
        MODEL_PARALLEL = False
elif MODEL == "resnet50":
    if DATASET == "imagenet":
        MODEL_FILE = None

# setup concept attack target index
CONCEPT_ATTACK_TARGET_NAME = args.concept_attack_target_name
CONCEPT_ATTACK_TARGET_INDEX = None
if args.concept_attack_target_name is not None:
    CONCEPT_ATTACK_TARGET_INDEX = label_name_to_index(
        CONCEPT_ATTACK_TARGET_CATEGORY, CONCEPT_ATTACK_TARGET_NAME, DATA_DIRECTORY
    )

# setup output directory
OUTPUT_FOLDER = f"{BASE_DIRECTORY}/output"
if args.experiment_name is not None:
    OUTPUT_FOLDER += f"/{args.experiment_name}"
if NOISE_ATTACK:
    OUTPUT_FOLDER = get_noise_attack_data_directory(
        OUTPUT_FOLDER,
        MODEL,
        DATASET,
        LAYER,
        NOISE_ATTACK_TYPE,
        NOISE_ATTACK_PROBABILITY,
        NOISE_ATTACK_STD,
        TIMESTAMP,
    )
elif CONCEPT_ATTACK:
    OUTPUT_FOLDER = get_concept_attack_data_directory(
        OUTPUT_FOLDER,
        MODEL,
        DATASET,
        LAYER,
        CONCEPT_ATTACK_UNIT,
        CONCEPT_ATTACK_TARGET_UNIT,
        CONCEPT_ATTACK_METHOD,
        CONCEPT_ATTACK_METHOD_PGD_EPSILON,
        CONCEPT_ATTACK_METHOD_PGD_STEPS,
        CONCEPT_ATTACK_TARGET_CATEGORY,
        CONCEPT_ATTACK_TARGET_NAME,
        TIMESTAMP,
    )
else:
    OUTPUT_FOLDER += f"/{MODEL}/{DATASET}"
    OUTPUT_FOLDER += f"/{LAYER}"
    OUTPUT_FOLDER += f"/baseline/{TIMESTAMP}"

# setup test mode, if provided, the test mode will be activated
if TEST_MODE:
    WORKERS = 1
    BATCH_SIZE = 4
    TALLY_BATCH_SIZE = 4
    TALLY_AHEAD = 1
    INDEX_FILE = "index_sm.csv"
    OUTPUT_FOLDER += "/test"
else:
    WORKERS = 32
    BATCH_SIZE = 64
    TALLY_BATCH_SIZE = 64
    TALLY_AHEAD = 64
    INDEX_FILE = "index.csv"

# create output directory
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# setup logger
logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.FileHandler(OUTPUT_FOLDER + "/log.txt"), logging.StreamHandler()],
)
logger = logging.getLogger("NetDissect")

# log all variables
logger.info(
    f" MODEL: {MODEL}, DATASET: {DATASET}, LAYER: {LAYER},"
    f" noise_attack: {NOISE_ATTACK}, NOISE_PROBABILITY:"
    f" {NOISE_ATTACK_PROBABILITY}, NOISE_ATTACK_STD: {NOISE_ATTACK_STD},"
    f" CONCEPT_ATTACK: {CONCEPT_ATTACK}, CONCEPT_ATTACK_UNIT: {CONCEPT_ATTACK_UNIT},"
    f" CONCEPT_ATTACK_LOSS_TYPE: {CONCEPT_ATTACK_LOSS_TYPE},"
    f" CONCEPT_ATTACK_METHOD: {CONCEPT_ATTACK_METHOD},"
    f" CONCEPT_ATTACK_METHOD_PGD_EPSILON: {CONCEPT_ATTACK_METHOD_PGD_EPSILON},"
    f" CONCEPT_ATTACK_METHOD_PGD_STEPS: {CONCEPT_ATTACK_METHOD_PGD_STEPS},"
    f" CONCEPT_ATTACK_TARGET_CATEGORY: {CONCEPT_ATTACK_TARGET_CATEGORY},"
    f" CONCEPT_ATTACK_TARGET_INDEX: {CONCEPT_ATTACK_TARGET_INDEX}, TEST_MODE:"
    f" {TEST_MODE}, WORKERS: {WORKERS}, BATCH_SIZE: {BATCH_SIZE}, TALLY_BATCH_SIZE:"
    f" {TALLY_BATCH_SIZE}, TALLY_AHEAD: {TALLY_AHEAD}, INDEX_FILE: {INDEX_FILE},"
    f" OUTPUT_FOLDER: {OUTPUT_FOLDER}"
    f" QUANTILE: {QUANTILE}"
)
