"""Entry point for datasets.

This also contains information about what models, loss functions and metric
functions should be used in association with a dataset.
"""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import torch

from . import cifar10
from . import epsilon
from . import metrics


DATASET_CHOICES = [
    "epsilon",
    "epsilon-small",
    "cifar10",
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

    elif name == "cifar10":
        train_dataset = cifar10.get_cifar10_dataset(train=True)
        test_dataset = cifar10.get_cifar10_dataset(train=False)
        model_class = cifar10.Cifar10CNN
        loss_fn = torch.nn.functional.cross_entropy
        metric_fns = {"accuracy": metrics.categorical_accuracy}

    else:
        raise ValueError(f"No dataset with name: {name}")

    return train_dataset, test_dataset, model_class, loss_fn, metric_fns
