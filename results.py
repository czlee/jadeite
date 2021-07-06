"""Result logging utilities.

Chuan-Zheng Lee <czlee@stanford.edu>
July 2021"""

import csv
import datetime
import json
import socket
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

    print("Saving results to: " + str(path))

    return path


def log_arguments(args, results_dir):

    info = {}
    info['script'] = sys.argv[0]
    info['started'] = datetime.datetime.now().isoformat()
    info['host'] = socket.gethostname()

    git = {}
    try:
        git['commit'] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except subprocess.CalledProcessError:
        print("\033[1;33mWarning: Could not detect git commit hash\033[0m")
        git['commit'] = None
        git['error'] = "could not detect commit"

    try:
        changed_files = subprocess.check_output(["git", "diff-index", "--name-only", "HEAD"])
    except subprocess.CalledProcessError:
        print("\033[1;33mWarning: Could not detect git diff-index\033[0m")
        git['changed_files'] = []
    else:
        git['changed_files'] = changed_files.decode().strip().split('\n')

    info['git'] = git
    info['args'] = vars(args)

    info['command'] = sys.argv

    with open(results_dir / "arguments.json", 'w') as f:
        json.dump(info, f, indent=2)


def log_evaluation(eval_dict, results_dir):
    filename = results_dir / "evaluation.json"
    eval_dict['finished'] = datetime.datetime.now().isoformat()
    with open(filename, 'w') as f:
        json.dump(eval_dict, f, indent=2)


class CsvLogger:
    """Similar to `tf.keras.callbacks.CsvLogger`, but to be called from
    tensorflow-federated iterative process loops. Also, doesn't implement the
    functionality we don't need."""

    def __init__(self, filename, index_field='epoch'):
        self.filename = filename
        self.index_field = index_field
        self.keys = None
        self.writer = None
        self.csv_file = open(filename, 'w')

    def _start_writer(self, metrics):
        if self.keys is None:
            self.keys = sorted(metrics.keys())

        fieldnames = [self.index_field, 'timestamp'] + self.keys
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

    def log(self, number, metrics):
        """Log a row. `metrics` should be a dict with the same keys as every
        other time this is called."""
        if self.writer is None:
            self._start_writer(metrics)
        now = datetime.datetime.now().isoformat()
        row_dict = {self.index_field: number, 'timestamp': now}
        row_dict.update(metrics)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def close(self):
        self.writer = None
        self.csv_file.close()


if __name__ == "__main__":
    # for debugging arguments.txt only
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-argument", default=42)
    parser.add_argument("--test-argument-2", default=False, action='store_true')
    args = parser.parse_args()

    results_dir = create_results_directory()
    log_arguments(args, results_dir)
