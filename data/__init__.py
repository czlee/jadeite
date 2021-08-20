"""Entry point for datasets.

This also contains information about what models, loss functions and metric
functions should be used in association with a dataset.
"""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import torch
import torchvision

from . import cifar10
from . import epsilon
from . import fashionmnist
from . import metrics
from . import resnet


DATASET_CHOICES = [
    "epsilon",
    "epsilon-small",
    "cifar10-simple",
    "cifar10-resnet20",
    "cifar10-resnet18",
    "cifar10-resnet20-flipcrop",
    "cifar10-resnet18-flipcrop",
    "fashionmnist-simple",
    "fashionmnist-cnn1",
    "fashionmnist-cnn2",
]


def get_datasets_etc(name):
    """Returns a 5-tuple
        (train_dataset, test_dataset, model_class, loss_fn, metric_fns)
    for use with the given dataset."""

    if name == "epsilon":
        train_dataset = epsilon.EpsilonDataset(train=True, small=False)
        test_dataset = epsilon.EpsilonDataset(train=False, small=False)
        model_class = epsilon.EpsilonLogisticModel
        loss_fn = torch.nn.functional.binary_cross_entropy
        metric_fns = {"accuracy": metrics.binary_accuracy}

    elif name == "epsilon-small":
        train_dataset = epsilon.EpsilonDataset(train=True, small=True)
        test_dataset = epsilon.EpsilonDataset(train=False, small=True)
        model_class = epsilon.EpsilonLogisticModel
        loss_fn = torch.nn.functional.binary_cross_entropy
        metric_fns = {"accuracy": metrics.binary_accuracy}

    elif name == "cifar10-simple":  # previous name: "cifar10"
        train_dataset = cifar10.get_cifar10_dataset(train=True, transform='norm1')
        test_dataset = cifar10.get_cifar10_dataset(train=False, transform='norm1')
        model_class = cifar10.Cifar10CNNSimple
        loss_fn = torch.nn.functional.cross_entropy
        metric_fns = {"accuracy": metrics.categorical_accuracy}

    elif name == "fashionmnist-simple":  # previous name: "fashionmnist"
        train_dataset = fashionmnist.get_fashion_mnist_dataset(train=True)
        test_dataset = fashionmnist.get_fashion_mnist_dataset(train=False)
        model_class = fashionmnist.FashionMnistNNSimple
        loss_fn = torch.nn.functional.cross_entropy
        metric_fns = {"accuracy": metrics.categorical_accuracy}

    elif name == "fashionmnist-cnn1":  # previous name: "fashionmnist-convnet"
        train_dataset = fashionmnist.get_fashion_mnist_dataset(train=True)
        test_dataset = fashionmnist.get_fashion_mnist_dataset(train=False)
        model_class = fashionmnist.FashionMnistCNN1
        loss_fn = torch.nn.functional.cross_entropy
        metric_fns = {"accuracy": metrics.categorical_accuracy}

    elif name == "fashionmnist-cnn2":
        train_dataset = fashionmnist.get_fashion_mnist_dataset(train=True)
        test_dataset = fashionmnist.get_fashion_mnist_dataset(train=False)
        model_class = fashionmnist.FashionMnistCNN2
        loss_fn = torch.nn.functional.cross_entropy
        metric_fns = {"accuracy": metrics.categorical_accuracy}

    elif name == "cifar10-resnet20":
        train_dataset = cifar10.get_cifar10_dataset(train=True, transform='norm1')
        test_dataset = cifar10.get_cifar10_dataset(train=False, transform='norm1')
        model_class = resnet.resnet20
        loss_fn = torch.nn.functional.cross_entropy
        metric_fns = {"accuracy": metrics.categorical_accuracy}

    elif name == "cifar10-resnet18":
        train_dataset = cifar10.get_cifar10_dataset(train=True, transform='norm1')
        test_dataset = cifar10.get_cifar10_dataset(train=False, transform='norm1')
        model_class = torchvision.models.resnet18
        loss_fn = torch.nn.functional.cross_entropy
        metric_fns = {"accuracy": metrics.categorical_accuracy}

    elif name == "cifar10-resnet20-flipcrop":
        train_dataset = cifar10.get_cifar10_dataset(train=True, transform='flip-crop-norm2')
        test_dataset = cifar10.get_cifar10_dataset(train=False, transform='norm2')
        model_class = resnet.resnet20
        loss_fn = torch.nn.functional.cross_entropy
        metric_fns = {"accuracy": metrics.categorical_accuracy}

    elif name == "cifar10-resnet18-flipcrop":
        train_dataset = cifar10.get_cifar10_dataset(train=True, transform='flip-crop-norm2')
        test_dataset = cifar10.get_cifar10_dataset(train=False, transform='norm2')
        model_class = torchvision.models.resnet18
        loss_fn = torch.nn.functional.cross_entropy
        metric_fns = {"accuracy": metrics.categorical_accuracy}

    else:
        raise ValueError(f"No dataset with name: {name}")

    return train_dataset, test_dataset, model_class, loss_fn, metric_fns
