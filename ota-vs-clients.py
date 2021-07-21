"""Messy implementation of repeating the over-the-air experiment and stepping
through a number of clients. Each experiment is saved to a new subdirectory of
the automatically created results directory."""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021


import argparse
import logging

import torch

import data.epsilon as epsilon
import metrics
import utils
from experiments import OverTheAirExperiment


parser = argparse.ArgumentParser(description=__doc__, conflict_handler='resolve')
OverTheAirExperiment.add_arguments(parser)
parser.add_argument("--small", action="store_true", default=False,
    help="Use a small dataset for testing")
parser.add_argument("-n", "--clients", type=int, nargs='+', default=[10],
    help="Number of clients, n")
parser.add_argument("-q", "--repeat", type=int, default=1,
    help="Number of times to repeat experiment")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s | %(message)s")
top_results_dir = utils.create_results_directory()
utils.log_arguments(args, top_results_dir)

train_dataset = epsilon.EpsilonDataset(train=True, small=args.small)
test_dataset = epsilon.EpsilonDataset(train=False, small=args.small)
loss_fn = torch.nn.functional.binary_cross_entropy
metric_fns = {"accuracy": metrics.binary_accuracy}

nclients_list = args.clients

for i in range(args.repeat):
    for n in nclients_list:
        args.clients = n
        print(f"=== Iteration {i} of {args.repeat}, {n} clients ===")
        results_dir = top_results_dir / f"clients-{n}-iteration-{i}"
        results_dir.mkdir()
        utils.log_arguments(args, results_dir, other_info={'iteration': i, 'clients': n})
        experiment = OverTheAirExperiment.from_arguments(
            train_dataset, test_dataset, epsilon.EpsilonLogisticModel,
            loss_fn, metric_fns, results_dir, args)
        experiment.run()
