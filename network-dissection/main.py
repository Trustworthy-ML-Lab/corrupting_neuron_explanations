import settings
from create_concept_attack_dataset import ConceptAdversaryDataset
from create_concept_attack_dataset_with_baseline import ConceptAdversaryDatasetBaseline
from create_noise_dataset import NoiseDataset
from feature_operation import FeatureOperator
from loader.model_loader import loadmodel
from visualize.report import generate_html_summary

logger = settings.logger

# load model
logger.info("Loading model")
model = loadmodel()

############ STEP 0: setup dataset ####################
if settings.NOISE_ATTACK:
    logger.info("Creating Noise Dataset")
    dataset = NoiseDataset(
        data_directory=settings.DATA_DIRECTORY,
        categories=settings.CATAGORIES,
        model=model,
        model_name=settings.MODEL,
        model_type=settings.MODEL_TYPE,
        dataset=settings.DATASET,
        layer=settings.LAYER,
        logger=logger,
        attack_type=settings.NOISE_ATTACK_TYPE,
        noise_level=settings.NOISE_ATTACK_STD,
        probability=settings.NOISE_ATTACK_PROBABILITY,
        batch_size=settings.BATCH_SIZE,
        timestamp=settings.TIMESTAMP,
    )
    dataset.create()
elif settings.CONCEPT_ATTACK and settings.CONCEPT_ATTACK_TARGET_UNIT:
    logger.info("Creating Concept Adversary Dataset with Baseline")
    fo = FeatureOperator(
        model, units=[settings.CONCEPT_ATTACK_UNIT, settings.CONCEPT_ATTACK_TARGET_UNIT]
    )
    thresholds, maxfeature = fo.quantile_threshold(
        savepath="quantile_baseline.npy",
        units=[settings.CONCEPT_ATTACK_UNIT, settings.CONCEPT_ATTACK_TARGET_UNIT],
    )
    dataset = ConceptAdversaryDatasetBaseline(
        data_directory=settings.DATA_DIRECTORY,
        categories=settings.CATAGORIES,
        model=model,
        model_name=settings.MODEL,
        model_type=settings.MODEL_TYPE,
        dataset=settings.DATASET,
        epsilon=settings.CONCEPT_ATTACK_METHOD_PGD_EPSILON,
        steps=settings.CONCEPT_ATTACK_METHOD_PGD_STEPS,
        method=settings.CONCEPT_ATTACK_METHOD,
        layer=settings.LAYER,
        source_unit=settings.CONCEPT_ATTACK_UNIT,
        target_unit=settings.CONCEPT_ATTACK_TARGET_UNIT,
        source_threshold=thresholds[0],
        target_threshold=thresholds[1],
        loss_type=settings.CONCEPT_ATTACK_LOSS_TYPE,
        batch_size=settings.BATCH_SIZE,
        timestamp=settings.TIMESTAMP,
    )
    dataset.create()
elif settings.CONCEPT_ATTACK:
    logger.info("Creating Concept Adversary Dataset")
    dataset = ConceptAdversaryDataset(
        data_directory=settings.DATA_DIRECTORY,
        categories=settings.CATAGORIES,
        model=model,
        model_name=settings.MODEL,
        model_type=settings.MODEL_TYPE,
        dataset=settings.DATASET,
        epsilon=settings.CONCEPT_ATTACK_METHOD_PGD_EPSILON,
        steps=settings.CONCEPT_ATTACK_METHOD_PGD_STEPS,
        method=settings.CONCEPT_ATTACK_METHOD,
        layer=settings.LAYER,
        unit=settings.CONCEPT_ATTACK_UNIT,
        loss_type=settings.CONCEPT_ATTACK_LOSS_TYPE,
        target_category=settings.CONCEPT_ATTACK_TARGET_CATEGORY,
        target_index=settings.CONCEPT_ATTACK_TARGET_INDEX,
        target_name=settings.CONCEPT_ATTACK_TARGET_NAME,
        batch_size=settings.BATCH_SIZE,
        timestamp=settings.TIMESTAMP,
    )
    dataset.create()
else:
    logger.info("Running with original dataset")


# create feature operator
logger.info("Creating Feature Operator")
fo = FeatureOperator(model)

############ STEP 1: calculating threshold ############
logger.info("STEP 1: calculating threshold")
thresholds, maxfeature = fo.quantile_threshold(savepath="quantile.npy")
maxfeature = maxfeature.detach().cpu().numpy()

############ STEP 2: calculating IoU scores ###########
logger.info("STEP 2: calculating IoU scores")
tally_result = fo.tally(thresholds, savepath="tally.csv")

############ STEP 3: generating results ###############
logger.info("STEP 3: generating results")
generate_html_summary(
    fo,
    settings.LAYER,
    tally_result=tally_result,
    maxfeature=maxfeature,
    thresholds=thresholds.detach().cpu().numpy(),
)
