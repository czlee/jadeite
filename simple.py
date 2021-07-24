"""Basic logistic regression."""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021


import argparse
import logging

import coloredlogs

import data
import utils
from experiments import SimpleExperiment

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
SimpleExperiment.add_arguments(parser)
parser.add_argument("-d", "--dataset", choices=data.DATASET_CHOICES, default='epsilon',
    help="Dataset (and associated model, loss and metric) to use")
args = parser.parse_args()

coloredlogs.install(level=logging.DEBUG, fmt="%(asctime)s %(levelname)s %(message)s", milliseconds=True)
results_dir = utils.create_results_directory()
utils.log_arguments(args, results_dir)

train_dataset, test_dataset, model_class, loss_fn, metric_fns = data.get_datasets_etc(args.dataset)
model = model_class()

experiment = SimpleExperiment.from_arguments(
    train_dataset, test_dataset, model, loss_fn, metric_fns, results_dir, args)
experiment.run()
