"""Helper module to read the epsilon dataset.

The epsilon dataset can be found at:
  https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#epsilon

In particular:

  https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2
  https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2

The simplest setup is to download this to the directory data/sources/epsilon
(using e.g. wget or curl) and extract it, and set DATA_DIRECTORY in config.py to
data/sources (with a fully qualified path)
"""

from pathlib import Path

import tensorflow as tf

try:
    from config import DATA_DIRECTORY
except ImportError:
    print("Copy config.py.example to config.py and set DATA_DIRECTORY to the path")
    print("where data files should be found.")
    exit(1)


epsilon_location = Path(DATA_DIRECTORY) / 'epsilon'


def _epsilon_dataset(filepath):

    def _epsilon_generator():
        try:
            f = open(filepath)  # must be opened inside generator
        except FileNotFoundError:
            _epsilon_hint()
            raise

        for line in f:
            parts = line.split()
            y = 1 if parts.pop(0) == '1' else 0
            x = [float(part.split(':')[1]) for part in parts]
            yield tf.constant(x), y

        f.close()

    return tf.data.Dataset.from_generator(
        _epsilon_generator,
        output_signature=(
            tf.TensorSpec(shape=(2000,), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ),
    )


def _epsilon_hint():
    print("-" * 80)
    print("\033[1;31mError: Data file not found\033[0m")
    print("Hint: Download the epsilon dataset (both files) from")
    print("  https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#epsilon")
    print("and extract it (using bunzip2) to this directory:")
    print("  " + str(epsilon_location))
    print("You can change this directory in config.py")
    print("-" * 80)


def _length_of_file(filepath):
    try:
        f = open(filepath)  # must be opened inside generator
    except FileNotFoundError:
        return None

    for i, _ in enumerate(f):
        pass
    f.close()

    return i + 1


def test_dataset():
    """Returns a `tf.data.Dataset` containing the test data from the "epsilon"
    dataset."""
    return _epsilon_dataset(epsilon_location / "epsilon_normalized.t")


def train_dataset():
    """Returns a `tf.data.Dataset` containing the training data from the
    "epsilon" dataset."""
    return _epsilon_dataset(epsilon_location / "epsilon_normalized")


ntrain = _length_of_file(epsilon_location / "epsilon_normalized")

ntest = _length_of_file(epsilon_location / "epsilon_normalized.t")
