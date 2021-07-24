"""Wrapper for the CIFAR-10 dataset, which is provided in the `torchvision`
package.

The model and data transform are taken directly from:
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import torch
import torch.nn as nn
import torchvision

try:
    from config import DATA_DIRECTORY
except ImportError:
    print("Copy config.py.example to config.py and set DATA_DIRECTORY to the path")
    print("where data files should be found.")
    exit(1)


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def get_cifar10_dataset(train=True):
    # We want it to download automatically, torchvision prints a message if it's
    # already downloaded, which is kind of annoying, so check if it's there
    # first and pass download=False if it is.
    download = not (DATA_DIRECTORY / "cifar-10-batches-py").exists()

    return torchvision.datasets.CIFAR10(root=DATA_DIRECTORY, train=train,
                                        download=download, transform=transform)


class Cifar10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
