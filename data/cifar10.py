"""Wrapper for the CIFAR-10 dataset, which is provided in the `torchvision`
package.

The model and data transform are taken directly from:
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from config import DATA_DIRECTORY


cifar10_transforms = {
    # copied from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    'norm1': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]),

    # copied from https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/trainer.py
    'flip-crop-norm2': torchvision.transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]),
    'norm2': torchvision.transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]),
}


def get_cifar10_dataset(train=True, transform='norm1'):
    # We want it to download automatically, torchvision prints a message if it's
    # already downloaded, which is kind of annoying, so check if it's there
    # first and pass download=False if it is.
    cifar10_directory = Path(DATA_DIRECTORY) / "cifar10"
    download = not (cifar10_directory / "cifar-10-batches-py").exists()

    return torchvision.datasets.CIFAR10(
        root=cifar10_directory,
        train=train,
        download=download,
        transform=cifar10_transforms[transform],
    )


# copied from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class Cifar10CNNSimple(nn.Module):

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
