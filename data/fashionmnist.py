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

from config import DATA_DIRECTORY


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
class FashionMnistNNSimple(nn.Module):

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


# This model is copied from the rTop-k repository, with only minor coding style edits.
# (see: https://arxiv.org/abs/2005.10761)
class FashionMnistCNN1(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


# copied from https://www.kaggle.com/pankajj/fashion-mnist-with-pytorch-93-accuracy
# and similar to
# https://colab.research.google.com/github/pytorch/ignite/blob/master/examples/notebooks/FashionMNIST.ipynb
class FashionMnistCNN2(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
