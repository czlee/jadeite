"""Shows the duration of runs in the given directories.

Convenience script to help with understanding runtime (for practical purposes,
not substantive research)."""

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
        return None
    return datetime.datetime.strptime(args['started'], isofmt)


def get_end_time(directory):
    evalfile = directory / "evaluation.json"
    try:
        with open(evalfile) as f:
            evaluation = json.load(f)
    except FileNotFoundError:
        return None
    return datetime.datetime.strptime(evaluation['finished'], isofmt)


def fmtdelta(duration):
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    hours += duration.days * 24
    if duration.microseconds >= 500000:
        seconds += 1
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def show_duration_statistics(directory, force=False):

    subdirectories = [d for d in directory.iterdir() if d.is_dir()]
    starts = [get_start_time(subd) for subd in subdirectories]
    finishes = [get_end_time(subd) for subd in subdirectories]

    data = sorted(zip(subdirectories, starts, finishes), key=lambda x: x[1])
    durations = []
    show_all_durations = force or len(data) <= 20

    for subd, start, finish in data:
        if start is None:
            duration_str = "no start time"
        elif finish is None:
            duration_str = "no end time"
        else:
            duration = finish - start
            durations.append(duration)
            duration_str = fmtdelta(duration)
        if show_all_durations:
            print(f" - {subd.name:37} {duration_str:>9}")

    if not show_all_durations:
        print(f"▶ {len(data)} experiments, not showing individually")

    if starts.count(None) + finishes.count(None) == 0:
        earliest_start = min(starts)
        latest_finish = max(finishes)
        total_duration = latest_finish - earliest_start
        print(f"\033[0;32mtotal duration:   {fmtdelta(total_duration):>10}\033[0m")

    if len(durations) > 0:
        average_duration = sum(durations, start=datetime.timedelta(0)) / len(durations)
        print(f"\033[0;32maverage duration: {fmtdelta(average_duration):>10}\033[0m")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directories", nargs='+', type=str,
    help="Directories, either in config.RESULTS_DIRECTORY or in their own right")
parser.add_argument("-f", "--force", action='store_true',
    help="Print all individual durations, even if there are more than 20")
args = parser.parse_args()

directories = get_directories_from_args(args.directories)

for directory in directories:
    print(f"\033[1;36m== {directory} ==\033[0m")
    show_status_line(directory)
    show_duration_statistics(directory, force=args.force)
