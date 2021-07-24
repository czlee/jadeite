"""Wrapper for the Fashion-MNIST dataset, which is provided in the `torchvision`
package.

The model and data transform are taken directly from:
    https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
"""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

from pathlib import Path

import torch.nn as nn
import torchvision

try:
    from config import DATA_DIRECTORY
except ImportError:
    print("Copy config.py.example to config.py and set DATA_DIRECTORY to the path")
    print("where data files should be found.")
    exit(1)


def get_fashion_mnist_dataset(train=True):
    # We want it to download automatically, torchvision prints a message if it's
    # already downloaded, which is kind of annoying, so check if it's there
    # first and pass download=False if it is.
    fashionmnist_directory = Path(DATA_DIRECTORY) / "fashion-mnist"
    download = not (fashionmnist_directory / "FashionMNIST" / "raw" / "t10k-labels-idx1-ubyte").exists()

    return torchvision.datasets.FashionMNIST(
        root=fashionmnist_directory,
        train=train,
        download=download,
        transform=torchvision.transforms.ToTensor(),
    )


# copied from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
class FashionMnistNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
