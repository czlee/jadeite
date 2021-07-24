"""Shows the duration of runs in the given directories.

Convenience script to help with understanding runtime (for practical uses, not
research uses)."""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import argparse
import datetime
import json

from list_results import show_status_line
from utils import get_directories_from_args


isofmt = '%Y-%m-%dT%H:%M:%S.%f'


def get_start_time(directory):
    argsfile = directory / "arguments.json"
    try:
        with open(argsfile) as f:
            args = json.load(f)
    except FileNotFoundError:
        print(f"{directory} doesn't have an arguments.json file")
        return None
    return datetime.datetime.strptime(args['started'], isofmt)


def get_end_time(directory):
    evalfile = directory / "evaluation.json"
    try:
        with open(evalfile) as f:
            evaluation = json.load(f)
    except FileNotFoundError:
        print(f"{directory} doesn't have an evaluation.json file")
        return None
    return datetime.datetime.strptime(evaluation['finished'], isofmt)


def get_duration(directory):
    return get_end_time(directory) - get_start_time(directory)


def show_duration_statistics(directory):

    subdirectories = [d for d in directory.iterdir() if d.is_dir()]
    subdirectories.sort()

    for subd in subdirectories:
        duration = get_duration(subd)
        print(f" - {subd.name:37} {duration}")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directories", nargs='+', type=str,
    help="Directories, either in config.RESULTS_DIRECTORY or in their own right")
args = parser.parse_args()

directories = get_directories_from_args(args.directories)

for directory in directories:
    print(f"\033[1;36m== {directory} ==\033[0m")
    show_status_line(directory)
    show_duration_statistics(directory)
