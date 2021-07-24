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
import itertools
import logging

import coloredlogs

import data
import utils


def run_experiments(experiment_class, description="Description not provided."):

    parser = argparse.ArgumentParser(
        description=description,
        conflict_handler='resolve',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    experiment_class.add_arguments(parser)
    parser.add_argument("-d", "--dataset", choices=data.DATASET_CHOICES, default='epsilon',
        help="Dataset (and associated model, loss and metric) to use")

    matrix_args = parser.add_argument_group("Experiment matrix arguments")
    matrix_args.add_argument("-n", "--clients", type=int, nargs='+', default=[10],
        help="Number of clients, n")
    matrix_args.add_argument("-N", "--noise", type=float, nargs='+', default=[1.0],
        help="Noise level (variance), σₙ²")
    matrix_args.add_argument("-q", "--repeat", type=int, default=1,
        help="Number of times to repeat experiment")
    args = parser.parse_args()

    coloredlogs.install(level=logging.DEBUG,
        fmt="%(asctime)s %(levelname)s %(message)s", milliseconds=True)
    top_results_dir = utils.create_results_directory()
    utils.log_arguments(args, top_results_dir)

    train_dataset, test_dataset, model_class, loss_fn, metric_fns = data.get_datasets_etc(args.dataset)

    nclients_list = args.clients
    noise_list = args.noise

    for i, clients, noise in itertools.product(range(args.repeat), nclients_list, noise_list):
        args.clients = clients
        args.noise = noise
        print(f"=== Iteration {i} of {args.repeat}, {clients} clients, noise {noise} ===")
        results_dir = top_results_dir / f"clients-{clients}-noise-{noise}-iteration-{i}"
        results_dir.mkdir()
        utils.log_arguments(args, results_dir,
            other_info={'iteration': i, 'clients': clients, 'noise': noise})
        experiment = experiment_class.from_arguments(
            train_dataset, test_dataset, model_class, loss_fn, metric_fns, results_dir, args)
        experiment.run()


if __name__ == "__main__":
    print("Use one of the convenience scripts, like dynpower.py or dynquant.py.")
