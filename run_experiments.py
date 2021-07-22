"""Main script for running long experiments.

Given an experiment class, this script repeats the experiment for each number of
clients specified with the `--clients` option, and repeats the experiments the
number of times specified in `--repeat`. Each experiment is saved to a new
subdirectory of the automatically created results directory.

This script isn't meant to be invoked directly. Instead, other scripts in this
directory reference the `run_experiments()` function in this module, passing
in the name of a class. This helps reduce boilerplate code, while retaining
clarity about which script does what.

Note: This class assumes that the class given has the form of a subclass of
`BaseFederatedExperiment`, i.e. that it takes the arguments that
`BaseFederatedExperiment.from_arguments()` takes. It doesn't work with
`SimpleExperiment`.
"""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021


import argparse
import logging

import coloredlogs
import torch

import data.epsilon as epsilon
import metrics
import utils


def run_experiments(experiment_class, description="Description not provided."):

    parser = argparse.ArgumentParser(
        description=description,
        conflict_handler='resolve',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    experiment_class.add_arguments(parser)
    parser.add_argument("--small", action="store_true", default=False,
        help="Use a small dataset for testing")
    parser.add_argument("-n", "--clients", type=int, nargs='+', default=[10],
        help="Number of clients, n")
    parser.add_argument("-q", "--repeat", type=int, default=1,
        help="Number of times to repeat experiment")
    args = parser.parse_args()

    coloredlogs.install(level=logging.DEBUG,
        fmt="%(asctime)s %(levelname)s %(message)s", milliseconds=True)
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
            experiment = experiment_class.from_arguments(
                train_dataset, test_dataset, epsilon.EpsilonLogisticModel,
                loss_fn, metric_fns, results_dir, args)
            experiment.run()


if __name__ == "__main__":
    print("Use one of the convenience scripts, like fedavg.py or overtheair.py.")
