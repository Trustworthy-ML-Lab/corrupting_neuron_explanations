# Corrupting Neuron Explanations of Deep Visual Features

This work studies robustness of Neuron Explanation Methods (NEMs) when images in probing dataset are corrupted by random and crafted perturbations.

This is the official repository for the paper [Corrupting Neuron Explanations of Deep Visual Features](https://openaccess.thecvf.com/content/ICCV2023/papers/Srivastava_Corrupting_Neuron_Explanations_of_Deep_Visual_Features_ICCV_2023_paper.pdf) published at ICCV'23. 

![attack_diagram](./attack-pipeline.gif)

# Abstract
The inability of DNNs to explain their black-box behavior has led to a recent surge of explainability methods. However, there are growing concerns that these explainability methods are not robust and trustworthy. In this work, we perform the first robustness analysis of Neuron Explanation Methods under a unified pipeline and show that these explanations can be significantly corrupted by random noises and well-designed perturbations added to their probing data. We find that even adding small random noise with a standard deviation of 0.02 can already change the assigned concepts of up to 28\% neurons in the deeper layers. Furthermore, we devise a novel corruption algorithm and show that our algorithm can manipulate the explanation of more than $80\%$ neurons by poisoning less than 10\% of probing data. This raises the concern of trusting Neuron Explanation Methods in real-life safety and fairness critical applications. 

# Network Dissection

## Installation
- Change to network-dissection folder
```
    cd network-dissection
```

- Install dependencies
```
    conda create -n netdissect python=3.6 -y
    conda activate netdissect
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
    pip3 install scipy==1.2.0 torchattacks
```

- Download the Broden dataset (~1GB space) and the example pretrained model. If you already download this, you can create a symbolic link to your original dataset.
```
    ./script/dlbroden.sh
```

## Quick Start

The entry of the code is [main.py](./network-dissection/main.py) and the input parameters are defined in [settings.py](./network-dissection/settings.py). Please refer to [demo.ipynb](./network-dissection/demo.ipynb) notebook for a quick start and [settings.py](./network-dissection/settings.py) for complete list of input parameters.

## Modifying the code

### Using your own model

1. The code for loading a model can be found in `./network-dissection/loader/model_loader.py`. You will need to modify `load_model` function. The function should return a `torch.nn.Module` object.
2. Run code with `--model` argument to specify the model name. For example, if you have a model named `my_model` added in `model_loader.py`, you can run the code with `--model my_model` argument.

### Using your own probing dataset
1. The code for loading a dataset can be found in `./network-dissection/loader/data_loader.py`. The classes to override are `SegmentationData` and `SegmentationPrefetcher`.

# MILAN

## Installation

- To run the code, set up a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.in
```

- Create `datsets` directory and download `places365` dataset

## Quick Start

The entry of the code is [attack.py](./milan/scripts/attack.py) and the input parameters are defined in the same file. Please refer to [demo.ipynb](./milan/demo.ipynb) notebook for a quick start. 

## Modifying the code

Please refer to original [MILAN](https://github.com/evandez/neuron-descriptions) repository for more details. Teh files related to attacking the probing images are defined in `./milan/src/attack` directory.

# Sources
This repository uses code from the following sources and we thank the authors for making their code publicly available:
1. Network Dissection: [https://github.com/CSAILVision/NetDissect-Lite](https://github.com/CSAILVision/NetDissect-Lite)
2. MILAN: [https://github.com/evandez/neuron-descriptions](https://github.com/evandez/neuron-descriptions)
3. PyTorch_CIFAR10: [https://github.com/huyvnphan/PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10)

# References
Please cite our paper if you find this repository useful.
```
@InProceedings{Srivastava_2023_ICCV,
    author    = {Srivastava, Divyansh and Oikarinen, Tuomas and Weng, Tsui-Wei},
    title     = {Corrupting Neuron Explanations of Deep Visual Features},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {1877-1886}
}
```