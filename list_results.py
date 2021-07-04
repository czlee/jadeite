"""Summarizes results in the given results directory."""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("dir", type=Path, nargs='?', default=Path("results"),
    help="Results directory")
args = parser.parse_args()

resultsdir = args.dir

if not resultsdir.is_dir():
    print(f"{resultsdir} is not a directory")
    exit(1)


children = sorted(resultsdir.iterdir())

for child in children:
    date = child.name

    argsfile = child / "arguments.txt"
    if not argsfile.exists():
        continue

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

        # arguments
        args = {}
        for line in f:
            key, value = line.split(sep=':', maxsplit=1)
            key = key.strip()
            value = value.strip()
            args[key] = value
        argsstring = " ".join(f"{key}={value}" for key, value in args.items())

        print(f"{date} {commit} {script:<20}  {argsstring}")
