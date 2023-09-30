import os

from PIL import Image
from torch.utils.data import Dataset


class ImageNetKaggle(Dataset):
    """
    Note: This dataset will not load the labels of the images.
    """

    def __init__(self, root, split, transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        # load path of all images
        self.samples = []
        folder = f"{self.root}/ILSVRC/Data/CLS-LOC/{split}"
        for filename in os.listdir(folder):
            self.samples.append(f"{folder}/{filename}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, -1
