"""Main script for running long experiments.

Given an experiment class, this script repeats the experiment for each number of
clients specified with the `--clients` option, and repeats the experiments the
number of times specified in `--repeat`. Each experiment is saved to a new
subdirectory of the automatically created results directory.

Note: This class assumes that the class given has the form of a subclass of
`BaseFederatedExperiment`, i.e. that it takes the arguments that
`BaseFederatedExperiment.from_arguments()` takes.
"""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021


import argparse
import itertools
import logging
import shutil
import textwrap

import coloredlogs

import data
import utils
from experiments import experiments_by_name


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


def make_top_level_description():
    """Constructs the description for the top-level parser. This lists each of
    the experiment classes with the short descriptions (hopefully) provided in
    the `description` attribute of the classes."""

    width, _ = shutil.get_terminal_size()

    intro = textwrap.dedent("""\
        Runs experiments. Provide one of the subcommands listed below.
        Use --help on subcommands to see help for that subcommand.
    """)
    lines = textwrap.wrap(intro, width=width)
    lines.extend(['', 'subcommands:'])

    max_name_length = max(len(name) for name in experiments_by_name.keys())
    indent = max_name_length + 2

    for name, experiment_class in experiments_by_name.items():
        text = name.ljust(indent) + textwrap.dedent(experiment_class.description)
        wrapped = textwrap.wrap(text, width=width,
            initial_indent="  ", subsequent_indent=" " * (indent + 2))
        lines.extend(wrapped)
        lines.append('')

    return "\n".join(lines)


def make_top_level_parser():
    """Constructs the top-level parser. This parser treats each different
    experiment class as a subcommand, collecting arguments from the experiment
    class."""

    parser = argparse.ArgumentParser(
        description=make_top_level_description(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='experiment', required=True)

    for name, experiment_class in experiments_by_name.items():
        subparser = subparsers.add_parser(name,
            description=experiment_class.description,
            conflict_handler='resolve',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        subparser.add_argument("-d", "--dataset", choices=data.DATASET_CHOICES, default='epsilon',
            metavar='DATASET',
            help="Dataset, model, loss and metric to use (despite the name, this argument "
                 "specifies all of them). Valid choices: " + ", ".join(data.DATASET_CHOICES))

        # Add all arguments for this experiment class
        experiment_class.add_arguments(subparser)

        # Override --clients and --noise, if present, with arguments allowing many values
        matrix_args = subparser.add_argument_group("Experiment matrix arguments")
        matrix_args.add_argument("-q", "--repeat", type=int, default=1,
            help="Number of times to repeat experiment")

        if 'clients' in experiment_class.default_params:
            matrix_args.add_argument("-n", "--clients", type=int, nargs='+', default=[10],
                help="Number of clients, n")

        if 'noise' in experiment_class.default_params:
            matrix_args.add_argument("-N", "--noise", type=float, nargs='+', default=[1.0],
                help="Noise level (variance), σₙ²")

    return parser


def run_experiments(args):

    experiment_class = experiments_by_name[args.experiment]

    coloredlogs.install(level=logging.DEBUG,
        fmt="%(asctime)s %(levelname)s %(message)s", milliseconds=True)
    top_results_dir = utils.create_results_directory()
    utils.log_arguments(args, top_results_dir)

    train_dataset, test_dataset, model_class, loss_fn, metric_fns = data.get_datasets_etc(args.dataset)

    matrix_labels = [label for label in all_matrix_labels if label in experiment_class.default_params]
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
    parser = make_top_level_parser()
    args = parser.parse_args()
    run_experiments(args)
