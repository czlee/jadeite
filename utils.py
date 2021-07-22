"""Result logging utilities."""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import csv
import datetime
import json
import logging
import os
import socket
import string
import subprocess
import sys
from pathlib import Path

from config import RESULTS_DIRECTORY

logger = logging.getLogger(__name__)
LATEST_SYMLINK = 'latest'


def create_results_directory(results_base_dir=RESULTS_DIRECTORY, latest_symlink=True, logfile=True):
    """Creates a timestamped results directory and returns a Path representing
    it.

    If `latest_symlink` is True, points the `latest` symlink to the new
    directory.

    If `logfile` is True, adds a FileHandler for 'output.log' in the new
    directory to the given logger.
    """
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(results_base_dir) / now
    if path.exists():
        for letter in string.ascii_lowercase:
            path = Path(results_base_dir) / (now + letter)
            if not path.exists():
                break
        else:
            logger.error("Could not create new results directory")
            exit(1)
    path.mkdir(parents=True)

    if latest_symlink:
        latest = Path('latest')
        if latest.is_symlink():
            latest.unlink()
        latest.symlink_to(path)

    if logfile:
        handler = logging.FileHandler(path / "output.log")
        formatter = logging.Formatter("[{levelname} {asctime} {name}] {message}", style='{')
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)

    logger.info("Saving results to: " + str(path))

    return path


def log_arguments(args, results_dir: Path, other_info=None):
    """Logs the arguments (presumably from the command line) to the file
    "arguments.json". This helps keep a full record of what the experiment was,
    in case there's later any doubt about what experiment runs represent, or
    their reliability.

    `results_dir` should be a `pathlib.Path` object reflecting where results for
    this experiment are being saved.

    If `other_info` is provided, it will be inserted into the file under the key
    "other".
    """
    info = {}
    info['script'] = sys.argv[0]
    info['started'] = datetime.datetime.now().isoformat()
    info['host'] = socket.gethostname()
    info['process_id'] = os.getpid()  # allows list_results.py to check if this is still running

    git = {}
    try:
        git['commit'] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except subprocess.CalledProcessError:
        logger.warning("Could not detect git commit hash")
        git['commit'] = None
        git['error'] = "could not detect commit"

    try:
        changed_files = subprocess.check_output(["git", "diff-index", "--name-only", "HEAD"])
    except subprocess.CalledProcessError:
        logger.warning("Could not detect git diff-index")
        git['changed_files'] = []
    else:
        changed_files_output = changed_files.decode().strip()
        git['changed_files'] = changed_files_output.split('\n') if changed_files_output else []

    info['git'] = git
    info['args'] = vars(args)
    if other_info is not None:
        info['other'] = other_info

    info['command'] = sys.argv

    with open(results_dir / "arguments.json", 'w') as f:
        json.dump(info, f, indent=2)


def log_evaluation(eval_dict, results_dir: Path):
    """Logs the given `eval_dict` to the file `evaluation.json`. This function
    also adds a timestamp under the key "finished".

    `results_dir` should be a `pathlib.Path` object reflecting where results for
    this experiment are being saved.
    """
    filename = results_dir / "evaluation.json"
    eval_dict['finished'] = datetime.datetime.now().isoformat()
    with open(filename, 'w') as f:
        json.dump(eval_dict, f, indent=2)


class CsvLogger:
    """CSV logger specialised for our purposes. It takes dicts, and the first
    row will be headers.
    """

    def __init__(self, filename, index_field='epoch'):
        self.filename = filename
        self.index_field = index_field
        self.keys = None
        self.writer = None
        self.csv_file = open(filename, 'w', newline='')

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


def divide_integer_evenly(n, m):
    """Returns a list of `m` integers summing to `n`, with elements as even as
    possible. For example:
    ```
        divide_integer_evenly(10, 4)  ->  [3, 3, 2, 2]
        divide_integer_evenly(20, 3)  ->  [7, 6, 6]
    ```
    """
    lengths = [n // m] * m
    for i in range(n - sum(lengths)):
        lengths[i] += 1
    return lengths
