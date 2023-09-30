import logging
from collections import OrderedDict
from turtle import pd

import torch
import torch.nn as nn
import torchvision
from torchvision.models import VGG
from torchvision.models.resnet import ResNet

import settings
from models.oldresnet152 import OldResNet152
from models.oldvgg16 import vgg16
from util.common import instrumented_layername

baseurl = "https://dissect.csail.mit.edu/models/"
model_factory = dict(vgg16=vgg16, resnet152=OldResNet152)
weights_filename = dict(
    alexnet="alexnet_places365-92864cf6.pth",
    vgg16="vgg16_places365-0bafbc55.pth",
    resnet152="resnet152_places365-f928166e5c.pth",
)
logger = logging.getLogger(__name__)


def shrink_to_layer(model, model_name, layer):
    """
    Shrink model to the layer
    """
    # get layer name
    features_name = instrumented_layername(model_name, layer)
    parts = features_name.split(".")

    # get layer list
    layers = OrderedDict()
    key_found = False
    for _, value in model._modules.items():
        # iterate through the layers
        for subkey, subvalue in value._modules.items():
            # add layer to the list
            layers[subkey] = subvalue

            # check if the layer is the target layer
            if subkey == parts[1]:
                key_found = True
                break

        if key_found:
            break

    # throw error if the layer is not found
    if not key_found:
        raise ValueError(f"Layer {parts[1]} not found in {model_name}")

    # create model uptil layer
    formatted_model = nn.Sequential(layers)

    return formatted_model


def loadmodel():
    if settings.MODEL == "vgg16" or settings.MODEL == "resnet152":
        # load model from local file
        model = model_factory[settings.MODEL](num_classes=settings.NUM_CLASSES)
        url = baseurl + weights_filename[settings.MODEL]
        sd = torch.hub.load_state_dict_from_url(url)
        model.load_state_dict(sd)
    elif settings.MODEL == "resnet50":
        # load model class
        model = torchvision.models.__dict__["resnet50"](pretrained=True)

        # add model parallelism
        model = torch.nn.DataParallel(model, device_ids=[0])

        # load model from local file
        # assume model file is in data parallel format
        if settings.MODEL_FILE:
            logger.info(f"Loading model from {settings.MODEL_FILE}")
            checkpoint = torch.load(settings.MODEL_FILE)
            model.load_state_dict(checkpoint["state_dict"])
        logging.info(f"Model loaded: {settings.MODEL}")
    else:
        if settings.MODEL_FILE is None:
            model = torchvision.models.__dict__[settings.MODEL](pretrained=True)
        else:
            checkpoint = torch.load(settings.MODEL_FILE)
            if (
                type(checkpoint).__name__ == "OrderedDict"
                or type(checkpoint).__name__ == "dict"
            ):
                model = torchvision.models.__dict__[settings.MODEL](
                    num_classes=settings.NUM_CLASSES
                )
                if settings.MODEL_PARALLEL:
                    state_dict = {
                        str.replace(k, "module.", ""): v
                        for k, v in checkpoint["state_dict"].items()
                    }  # the data parallel layer will add 'module' before each layer name
                else:
                    state_dict = checkpoint
                model.load_state_dict(state_dict)
            else:
                model = checkpoint

    model = shrink_to_layer(model, settings.MODEL, settings.LAYER)

    # set model to cuda
    model = model.cuda()

    # set model to eval mode
    model = model.eval()

    return model
