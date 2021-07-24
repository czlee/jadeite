"""Shows the command noted in the arguments.json file. This is just a very
simple convenience script to make up for how the command is stored as a list of
words (like, how they're understood by the command line) rather than something
you can copy and paste into bash."""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import argparse
import json

from utils import get_directories_from_args


def show_command(directory):
    argsfile = directory / "arguments.json"
    if not argsfile.exists():
        print(f"{directory} doesn't have an arguments.json file")
        return

    with open(argsfile) as f:
        args = json.load(f)

    print(f"== command for {directory} ==")
    command = ["python"] + args['command']
    print(" ".join(command))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directories", nargs='+', type=str,
    help="Directories, either in config.RESULTS_DIRECTORY or in their own right")
args = parser.parse_args()

directories = get_directories_from_args(args.directories)

for directory in directories:
    show_command(directory)
