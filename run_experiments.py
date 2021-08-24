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


logger = logging.getLogger(__name__)


# This defines the labels that are used in the experiment matrix, except for
# 'iteration'/'--repeat', which is handled specially (because it is not an
# experiment parameter and common to all experiment types). The main function
# will filter this down for just those arguments relevant to the
# `experiment_class`. If you add another label to this matrix, the other thing
# you must do is add a `matrix_args.add_argument(...)` line below, conditioned
# on the label being in `matrix_labels`.
all_matrix_labels = ['clients', 'noise']


def write_nrepeats_to_file(results_dir, nrepeats):
    with open(results_dir / "repeats", 'w') as f:
        f.write(f"{nrepeats:d}\n")


def check_if_finished(results_dir, iteration):
    try:
        with open(results_dir / "repeats", 'r') as f:
            contents = f.readline()
    except FileNotFoundError:
        logger.error("Could not found 'repeats' file, continuing...")
        return False

    try:
        nrepeats = int(contents)
    except ValueError:
        logger.error(f"Invalid 'repeats' file first line: {contents.strip()!r}, continuing...")
        return False

    if iteration >= nrepeats:
        logger.info(f"Iteration {iteration} >= nrepeats {nrepeats}, stopping.")
        return True

    return False


def check_immediate_stop(results_dir):
    return (results_dir / "stop-now").exists()


def run_experiments(experiment_class, description="Description not provided."):

    matrix_labels = [label for label in all_matrix_labels if label in experiment_class.default_params]

    parser = argparse.ArgumentParser(
        description=description,
        conflict_handler='resolve',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    experiment_class.add_arguments(parser)
    parser.add_argument("-d", "--dataset", choices=data.DATASET_CHOICES, default='epsilon',
        metavar='DATASET',
        help="Dataset, model, loss and metric to use (despite the name, this argument "
             "specifies all of them). Valid choices: " + ", ".join(data.DATASET_CHOICES))

    matrix_args = parser.add_argument_group("Experiment matrix arguments")
    matrix_args.add_argument("-q", "--repeat", type=int, default=1,
        help="Number of times to repeat experiment")

    if 'clients' in matrix_labels:
        matrix_args.add_argument("-n", "--clients", type=int, nargs='+', default=[10],
            help="Number of clients, n")

    if 'noise' in matrix_labels:
        matrix_args.add_argument("-N", "--noise", type=float, nargs='+', default=[1.0],
            help="Noise level (variance), σₙ²")

    args = parser.parse_args()

    coloredlogs.install(level=logging.DEBUG,
        fmt="%(asctime)s %(levelname)s %(message)s", milliseconds=True)
    top_results_dir = utils.create_results_directory()
    utils.log_arguments(args, top_results_dir)

    train_dataset, test_dataset, model_class, loss_fn, metric_fns = data.get_datasets_etc(args.dataset)

    matrix = [getattr(args, label).copy() for label in matrix_labels]

    write_nrepeats_to_file(top_results_dir, args.repeat)
    iteration = 0

    while not check_if_finished(top_results_dir, iteration):

        for matrix_values in itertools.product(*matrix):

            matrix_dict = {}
            title_strs = []
            dirname_strs = []

            for label, value in zip(matrix_labels, matrix_values):
                setattr(args, label, value)
                matrix_dict[label] = value
                title_strs.append(f"{label} {value}")
                dirname_strs.append(f"{label}-{value}")

            matrix_dict['iteration'] = iteration

            logging.info("=== " + ", ".join(f"{lab} {val}" for lab, val in matrix_dict.items()) + " ===")
            results_dir = top_results_dir / "-".join(f"{lab}-{val}" for lab, val in matrix_dict.items())
            results_dir.mkdir()
            utils.log_arguments(args, results_dir, other_info=matrix_dict)

            experiment = experiment_class.from_arguments(
                train_dataset, test_dataset, model_class, loss_fn, metric_fns, results_dir, args)
            experiment.run()

            if check_immediate_stop(top_results_dir):
                logger.warning("'stop-now' file found, stopping immediately!")
                exit()

        iteration += 1


if __name__ == "__main__":
    print("Use one of the convenience scripts, like dynpower.py or dynquant.py.")
