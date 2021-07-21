"""Basic logistic regression using federated averaging."""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021


import argparse
import logging

import coloredlogs
import torch

import data.epsilon as epsilon
import metrics
import utils
from experiments import FederatedAveragingExperiment


parser = argparse.ArgumentParser(description=__doc__)
FederatedAveragingExperiment.add_arguments(parser)
parser.add_argument("--small", action="store_true", default=False,
    help="Use a small dataset for testing")
args = parser.parse_args()

coloredlogs.install(level=logging.DEBUG, fmt="%(asctime)s %(levelname)s %(message)s", milliseconds=True)
results_dir = utils.create_results_directory()
utils.log_arguments(args, results_dir)

train_dataset = epsilon.EpsilonDataset(train=True, small=args.small)
test_dataset = epsilon.EpsilonDataset(train=False, small=args.small)
loss_fn = torch.nn.functional.binary_cross_entropy
metric_fns = {"accuracy": metrics.binary_accuracy}

experiment = FederatedAveragingExperiment.from_arguments(
    train_dataset, test_dataset, epsilon.EpsilonLogisticModel,
    loss_fn, metric_fns, results_dir, args)
experiment.run()
