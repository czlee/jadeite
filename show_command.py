"""Shows the command noted in the arguments.json file."""

# Chuan-Zheng Le
# July 2021

import argparse
import json
import pathlib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directories", nargs='+', type=pathlib.Path)
args = parser.parse_args()

for directory in args.directories:
    if not directory.is_dir():
        print(f"{directory} is not a directory")
        continue
    argsfile = directory / "arguments.json"
    if not argsfile.exists():
        print(f"{directory} doesn't have an arguments.json file")
        continue
    with open(argsfile) as f:
        args = json.load(f)
    print(f"== command for {directory} ==")
    print(" ".join(args['command']))
