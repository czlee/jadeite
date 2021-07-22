"""Lists result directories (including subdirectories in composite directories)
that appear to be unfinished, and also aren't currently running. Intended to
assist with result directory cleanup. The output can be piped to `rm -r` to
quickly delete unfinished experiments using `xargs`, like this:

  $ python list_unfinished.py | xargs rm -r

but be sure to inspect the output of `python list_unfinished.py` by itself
before committing to deleting files!
"""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import argparse
from pathlib import Path

import psutil

from list_results import (detect_composite_status, has_finished, is_composite_directory,
                          process_arguments, process_legacy_arguments)


def print_if_unfinished(path):
    if not path.is_dir():
        return
    if not has_finished(path):
        print(path)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("dir", type=Path, nargs='?', default=Path("results"),
    help="Results directory")
args = parser.parse_args()

resultsdir = args.dir

if not resultsdir.is_dir():
    print(f"{resultsdir} is not a directory")
    exit(1)

directories = sorted(resultsdir.iterdir())

for directory in directories:
    if not directory.is_dir():
        continue

    if (directory / "arguments.json").exists():
        info_tuple = process_arguments(directory / "arguments.json")
    elif (directory / "arguments.txt").exists():
        info_tuple = process_legacy_arguments(directory / "arguments.txt")
    else:
        continue

    _, _, _, process_id, arguments = info_tuple

    if process_id is not None and psutil.pid_exists(process_id):
        continue

    if is_composite_directory(arguments):
        for child in sorted(directory.iterdir()):
            print_if_unfinished(child)

        unfinished, finished, expected = detect_composite_status(directory, arguments)
        if unfinished == 0 and finished == 0:
            print(directory)

    else:
        print_if_unfinished(directory)
