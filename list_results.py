"""Lists results directories with a summary of their contents.
Useful for quickly inspecting which directory contains which experiments.
"""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import argparse
import datetime
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("dir", type=Path, nargs='?', default=Path("results"),
    help="Results directory")
parser.add_argument("-s", "--show", nargs='+', default=[],
    help="Always show these arguments, even if equal to the default")
when = parser.add_mutually_exclusive_group()
when.add_argument("-r", "--recent", type=str, nargs='?', default=None, const='1d',
    help="Only show directories less than a day old, or less than a specified time, "
         "e.g. 2d for 2 days, 3h for 3 hours, 1d5h for 1 day 5 hours")
when.add_argument("-a", "--after", type=str, default=None,
    help="Only show directories after this date, specified in the format "
         "yyyymmdd-hhmmss, partial specifications (e.g. yyyymmdd-hh) allowed")
args = parser.parse_args()

DEFAULT_ARGUMENTS = {
    'rounds': 20,
    'batch_size': 64,
    'clients': 10,
    'lr_client': 0.01,
    'noise': 1.0,
    'power': 1.0,
    'parameter_radius': 1.0,
    'small': False,
    'cpu': False,
}

resultsdir = args.dir

if not resultsdir.is_dir():
    print(f"{resultsdir} is not a directory")
    exit(1)


def process_legacy_arguments(argsfile):
    """Process arguments that were in the old arguments.txt format."""
    with open(argsfile) as f:
        script = None
        started = None
        commit = None
        for line in f:
            if line.startswith("script: "):
                script = line[8:].strip()
            if line.startswith("started: "):
                started = line[9:].strip()
                started = datetime.datetime.strptime(started, '%Y-%m-%dT%H:%M:%S.%f')
            if line.startswith("commit: "):
                commit = line[8:15].strip()
            if line.strip() == "== arguments ==":
                break

        arguments = {}
        for line in f:
            key, value = line.split(sep=':', maxsplit=1)
            key = key.strip()
            value = value.strip()
            arguments[key] = value

    return script, started, commit, arguments


def process_arguments(argsfile):
    """Process arguments in the standard arguments.json format."""
    with open(argsfile) as f:
        args_dict = json.load(f)
    script = args_dict.get('script', '???')
    started = args_dict.get('started', None)
    if started is not None:
        started = datetime.datetime.strptime(started, '%Y-%m-%dT%H:%M:%S.%f')
    commit = args_dict.get('git', {}).get('commit', '???    ')[:7]
    arguments = args_dict.get('args', {})
    return script, started, commit, arguments


def shorten_key_name(key):
    """Shortens long keys."""
    words = key.split("_")
    if len(words) == 1:
        return key
    return "".join(word[0] for word in words).upper()


def format_arg_value(value):
    """Formats an argument value for display.
    Currently, this just simplifies lists of consecutive integers."""
    if not isinstance(value, list):
        return str(value)
    if len(value) == 1:
        return str(value)
    if not all([isinstance(x, int) for x in value]):
        return str(value)

    lowest = min(value)
    highest = max(value)
    if value == list(range(lowest, highest + 1)):
        return f"{{{lowest}..{highest}}}"
    return str(value)


def has_started(directory):
    """Returns True if the experiments in this directory appear to have
    (non-trivially) started.
    """
    return len(list(directory.iterdir())) > 1


def has_finished(directory):
    """Returns True if the experiments in this directory appear to have
    finished.
    """
    if (directory / "result.txt").exists():
        return True
    if (directory / "evaluation.json").exists():
        return True


def parse_start_time(args):
    if args.recent is not None:
        pattern = r'^(?:(\d(?:\.\d)?)d)?(?:(\d(?:\.\d)?)h)?$'
        m = re.match(pattern, args.recent)
        if not m:
            print(f"Invalid --recent argument: {args.recent}")
            exit(1)
        days, hours = m.groups()
        period = datetime.timedelta(days=float(days) if days else 0, hours=float(hours) if hours else 0)
        return datetime.datetime.now() - period

    if args.after is not None:
        formats = ["%Y", "%Y%m", "%Y%m%d", "%Y%m%d-%H", "%Y%m%d-%H%M", "%Y%m%d-%H%M%S"]
        for fmt in formats:
            try:
                start_time = datetime.datetime.strptime(args.after, fmt)
            except ValueError:
                continue
            else:
                return start_time
        else:
            print(f"Invalid --after argument: {args.after}")
            exit(1)

    return None


children = sorted(resultsdir.iterdir())
start_time = parse_start_time(args)
if start_time:
    print("\033[1;36mShowing directories after:", start_time.isoformat(), "\033[0m")

for child in children:
    if not child.is_dir():
        continue

    date = child.name

    if (child / "arguments.json").exists():
        script, started, commit, arguments = process_arguments(child / "arguments.json")
    elif (child / "arguments.txt").exists():
        script, started, commit, arguments = process_legacy_arguments(child / "arguments.txt")
    else:
        if start_time and datetime.datetime.strptime(date, "%Y%m%d-%H%M%S") > start_time:
            print(f"\033[1;31m{date}         ???\033[0m")
        continue

    if start_time and started and started < start_time:
        continue

    formatted = {
        shorten_key_name(key): format_arg_value(value)
        for key, value in arguments.items()
        if DEFAULT_ARGUMENTS.get(key) != value or key in args.show
    }

    argsstring = " ".join(f"\033[0;90m{key}=\033[0m{value}" for key, value in formatted.items())

    if not has_started(child):
        script = "(?) " + script
        color = "\033[0;31m"
    elif not has_finished(child):
        script = "(*) " + script
        color = "\033[0;32m"
    else:
        color = ""

    print(f"{color}{date} {commit} {script:<21}\033[0m  {argsstring}")
