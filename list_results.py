"""Summarizes results in the given results directory."""
# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("dir", type=Path, nargs='?', default=Path("results"),
    help="Results directory")
parser.add_argument("-s", "--show", nargs='+', default=[],
    help="Always show these arguments, even if equal to the default")
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
}

resultsdir = args.dir

if not resultsdir.is_dir():
    print(f"{resultsdir} is not a directory")
    exit(1)


def process_legacy_arguments(argsfile):
    with open(argsfile) as f:
        script = None
        commit = None
        for line in f:
            if line.startswith("script: "):
                script = line[8:].strip()
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

    return script, commit, arguments


children = sorted(resultsdir.iterdir())

for child in children:
    date = child.name

    argsfile = child / "arguments.json"

    if not argsfile.exists():
        legacy_argsfile = child / "arguments.txt"
        if legacy_argsfile.exists():
            script, commit, arguments = process_legacy_arguments(legacy_argsfile)
        else:
            print(f"\033[1;31m{date}         ???\033[0m")
            continue

    else:
        with open(argsfile) as f:
            args_dict = json.load(f)
        script = args_dict.get('script', '???')
        commit = args_dict.get('git', {}).get('commit', '???    ')[:7]
        arguments = args_dict.get('args', {})

    argsstring = " ".join(f"{key}={value}" for key, value in arguments.items()
                          if DEFAULT_ARGUMENTS.get(key) != value or key in args.show)

    # is it done? did it even start?
    if len(list(child.iterdir())) == 1:
        script = "(?) " + script
        color = "\033[0;31m"
    elif not (child / "result.txt").exists():
        script = "(*) " + script
        color = "\033[0;32m"
    else:
        color = ""

    print(f"{color}{date} {commit} {script:<21}  {argsstring}\033[0m")
