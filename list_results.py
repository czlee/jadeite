#!/usr/bin/env python
"""Lists results directories with a summary of their contents.
Useful for quickly inspecting which directory contains which experiments.
Also shows if any scripts are still running.

For scripts that run many experiments, it shows the number of experiments
finished, expected and unfinished. For example, "13/20  r1" means 13 of the
expected 20 experiments have finished, 1 is unfinished, and the relevant process
appears still to be running ("r") on this machine. The other status codes are
"u" for unfinished but not running, and "?" for hasn't started.
"""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import argparse
import datetime
import itertools
import json
import re
from pathlib import Path

import psutil

from config import RESULTS_DIRECTORY

DEFAULT_ARGUMENTS = {
    'rounds': 20,
    'batch_size': 64,
    'clients': 10,
    'lr_client': 0.01,
    'momentum_client': 0.0,
    'noise': 1.0,
    'power': 1.0,
    'parameter_radius': 1.0,
    'small': False,
    'cpu': False,
    'epochs': 1,
    'ema_coefficient': 1 / 3,
    'power_update_period': 1,
    'power_quantile': 0.9,
    'power_factor': 0.9,
    'qrange_update_period': 1,
    'qrange_param_quantile': 0.9,
    'qrange_client_quantile': 0.9,
    'rounding_method': 'stochastic',
    'zero_bits_strategy': 'read-zero',
    'save_models': True,
}


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

    return script, started, commit, None, arguments


def process_arguments(argsfile):
    """Process arguments in the standard arguments.json format."""
    with open(argsfile) as f:
        args_dict = json.load(f)
    script = args_dict.get('script', '???')
    started = args_dict.get('started', None)
    if started is not None:
        started = datetime.datetime.strptime(started, '%Y-%m-%dT%H:%M:%S.%f')
    git = args_dict.get('git', {})
    commit = git.get('commit', '???    ')[:7]
    changed_files = git.get('changed_files', [])
    if changed_files and changed_files != ['']:
        commit += "*"
    else:
        commit += " "
    process_id = args_dict.get('process_id')
    arguments = args_dict.get('args', {})
    return script, started, commit, process_id, arguments


def is_composite_argument(key, value):
    if key == 'repeat':
        return True
    if key in ['clients', 'noise'] and isinstance(value, list):
        return True
    return False


def format_args_string(arguments, always_show=[]):
    args_items = sorted(arguments.items(), key=lambda item: not is_composite_argument(*item))
    formatted_args = [
        format_argument(key, value) for key, value in args_items
        if DEFAULT_ARGUMENTS.get(key) != value or key in always_show
    ]
    return " ".join(formatted_args)


def format_argument(key, value):
    is_composite = is_composite_argument(key, value)
    key = shorten_key_name(key)
    value = format_arg_value(value)
    if is_composite:
        return f"\033[1;36m{key}=\033[0;36m{value}\033[0m"
    else:
        return f"\033[0;90m{key}=\033[0m{value}"


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
    if args.all:
        return None

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

    if args.recent is not None:
        pattern = r'^(?:(\d+(?:\.\d*)?)d)?(?:(\d+(?:\.\d*)?)h)?$'
        m = re.match(pattern, args.recent)
        if not m:
            print(f"Invalid --recent argument: {args.recent}")
            exit(1)
        days, hours = m.groups()
        period = datetime.timedelta(days=float(days) if days else 0, hours=float(hours) if hours else 0)
        return datetime.datetime.now() - period

    return None


def is_composite_directory(arguments):
    if not arguments:
        return False
    if isinstance(arguments.get('clients', None), list):
        return True
    return False


def detect_composite_status(directory, arguments):
    """Returns a 3-tuple `(unfinished, finished, expected)` where
    - `unfinished` is the number of directories that exist but have not finished
    - `finished` is the number of directories that have finished

    It is assumed that the directory is composite, i.e. that
    `is_composite_directory(directory)` is true.
    """
    clients = arguments['clients']
    noise = arguments.get('noise')
    repeat = arguments['repeat']
    finished = 0
    unfinished = 0

    matrix = (range(repeat), clients)
    expected = repeat * len(clients)
    childname_template = "clients-{1}-iteration-{0}"
    if isinstance(noise, list):
        matrix += (noise,)
        expected *= len(noise)
        childname_template = "clients-{1}-noise-{2}-iteration-{0}"

    for values in itertools.product(*matrix):
        childname = childname_template.format(*values)
        if has_finished(directory / childname):
            finished += 1
        elif (directory / childname).exists():
            unfinished += 1

    return unfinished, finished, expected


def show_status_line(directory, if_after=None, always_show=[]):
    """Shows the status and arguments of the directory in a single line.

    If `if_after` is provided, it should be a `datetime.datetime` object, and if
    the directory indicates that its experiments started after this time, this
    does nothing.

    Arguments in `always_show` are always shown, even if they match the default.
    """

    dirname = directory.name

    if (directory / "arguments.json").exists():
        info_tuple = process_arguments(directory / "arguments.json")
    elif (directory / "arguments.txt").exists():
        info_tuple = process_legacy_arguments(directory / "arguments.txt")
    else:
        if if_after and datetime.datetime.strptime(dirname, "%Y%m%d-%H%M%S") > if_after:
            print(f"\033[1;31m{dirname}         ???\033[0m")
        return

    script, started, commit, process_id, arguments = info_tuple

    if if_after and started and started < if_after:
        return

    argsstring = format_args_string(arguments, always_show=always_show)
    is_running = process_id is not None and psutil.pid_exists(process_id)

    if is_composite_directory(arguments):
        unfinished, finished, expected = detect_composite_status(directory, arguments)
        if is_running:
            color = "\033[0;32m"
            status = f"{finished:>3}/{expected:<3} r{unfinished}"
        elif unfinished > 0:
            # not running, but at least one is unfinished
            color = "\033[1;35m"
            status = f"{finished:>3}/{expected:<3} u{unfinished}"
        elif finished == expected:
            # seems to be done and dusted
            color = "\033[1;36m"
            status = f"{finished:>3}/{expected:<3}   "
        else:
            # seems to be incomplete but no single run is unfinished
            color = "\033[1;33m"
            status = f"{finished:>3}/{expected:<3} u{unfinished}"
    elif is_running:
        status = "        r  "
        color = "\033[0;32m"
    elif not has_started(directory):
        status = "        ?  "
        color = "\033[0;31m"
    elif not has_finished(directory):
        status = "        u  "
        color = "\033[0;33m"
    else:
        status = ""
        color = ""

    print(f"{color}{dirname:16} {commit} {status:10} {script:<17}\033[0m  {argsstring}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dir", type=Path, nargs='?', default=RESULTS_DIRECTORY,
        help=f"Results directory (default: {RESULTS_DIRECTORY})")
    parser.add_argument("-s", "--show", nargs='+', default=[], metavar="ARG",
        help="Always show these arguments, even if equal to the default")
    when = parser.add_mutually_exclusive_group()
    when.add_argument("-r", "--recent", type=str, default='1d', metavar="PERIOD",
        help="Only show directories less than a day old, or less than a specified "
             "period of time, e.g. 2d for 2 days, 3h for 3 hours, 1d5h for 1 day "
             "5 hours. This is the default, with a period of 1 day.")
    when.add_argument("-a", "--after", type=str, default=None, metavar="DATETIME",
        help="Only show directories after this date and time, specified in the format "
             "yyyymmdd-hhmmss, partial specifications (e.g. yyyymmdd-hh) allowed")
    when.add_argument("-A", "--all", action="store_true", default=False,
        help="List all directories, no matter how old")
    args = parser.parse_args()

    resultsdir = args.dir
    if not resultsdir.is_dir():
        print(f"{resultsdir} is not a directory")
        exit(1)

    directories = sorted(resultsdir.iterdir())
    start_time = parse_start_time(args)
    if start_time:
        print("\033[1;37mShowing directories after:", start_time.isoformat(), "\033[0m")

    for directory in directories:
        if not directory.is_dir():
            continue
        show_status_line(directory, if_after=start_time, always_show=args.show)
