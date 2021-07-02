"""Result logging utilities"""

import datetime
import subprocess
import sys
from pathlib import Path

from config import RESULTS_DIRECTORY


LATEST_SYMLINK = 'latest'


def create_results_directory(results_base_dir=RESULTS_DIRECTORY):
    """Creates a timestamped results directory and returns a Path representing
    it."""
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(results_base_dir) / now
    path.mkdir(parents=True)

    latest = Path('latest')
    if latest.is_symlink():
        latest.unlink()
    latest.symlink_to(path)

    return path


def log_arguments(args, results_dir):
    filename = results_dir / "arguments.txt"
    with open(filename, 'w') as f:
        f.write("script: " + sys.argv[0] + "\n")

        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        f.write("commit: " + commit + "\n")

        f.write("\n== arguments ==\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")


def log_evaluation(eval_dict, results_dir):
    filename = results_dir / "result.txt"
    with open(filename, 'w') as f:
        for key, value in eval_dict.items():
            f.write(f"{key}: {value}\n")
