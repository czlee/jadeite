"""Metrics not built in to PyTorch."""

import torch


def binary_accuracy(pred, y):
    return ((pred > 0.5) == y).type(torch.float).mean()


def categorical_accuracy(pred, y):
    _, guesses = torch.max(pred, dim=1)
    return (guesses == y).type(torch.float).mean()
