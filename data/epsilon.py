"""Helper module to read the epsilon dataset.

The epsilon dataset can be found at:
  https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#epsilon

In particular:

  https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2
  https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2

The simplest setup is to download this to the directory data/sources/epsilon
(using e.g. wget or curl) and extract it, and set DATA_DIRECTORY in config.py to
data/sources (with a fully qualified path).
"""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021


from pathlib import Path

import torch

try:
    from config import DATA_DIRECTORY
except ImportError:
    print("Copy config.py.example to config.py and set DATA_DIRECTORY to the path")
    print("where data files should be found.")
    exit(1)


def _get_data_location(train, small):
    directory = 'epsilon-small' if small else 'epsilon'
    filename = 'epsilon_normalized' if train else 'epsilon_normalized.t'
    return Path(DATA_DIRECTORY) / directory / filename


class EpsilonDataset(torch.utils.data.TensorDataset):
    """Loads the epsilon dataset into memory, and accesses as a TensorDataset."""

    def __init__(self, train=True, small=False, verbose=True):
        self.filename = _get_data_location(train, small)
        self.verbose = verbose
        tensors = self._get_tensors(self.filename)
        super().__init__(*tensors)

    def _verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _get_tensors(self, filename):
        try:
            f = open(filename)  # must be opened inside generator
        except FileNotFoundError:
            _epsilon_hint(filename)
            raise

        x = []
        y = []
        self._verbose(f"Loading data from {self.filename}...", end=' ', flush=True)

        for i, line in enumerate(f):
            if i % 200 == 0:
                self._verbose(f"\rLoading data from {self.filename}, up to "
                              f"line {i}...", end=' ', flush=True)
            parts = line.split()
            y.append([1.0 if parts.pop(0) == '1' else 0.0])
            x.append([float(part.split(':')[1]) for part in parts])

        self._verbose("done.")
        f.close()

        return (torch.tensor(x), torch.tensor(y))


class EpsilonIterableDataset(torch.utils.data.IterableDataset):
    """Like EpsilonDataset, but streams from the file."""

    def __init__(self, train=True, small=True):
        self.filename = _get_data_location(train, small)
        self.length = _length_of_file(self.filename)

    def __iter__(self):
        def _epsilon_generator():
            try:
                f = open(self.filename)  # must be opened inside generator
            except FileNotFoundError:
                _epsilon_hint(self.filename)
                raise

            for line in f:
                parts = line.split()
                y = 1.0 if parts.pop(0) == '1' else 0.0
                x = [float(part.split(':')[1]) for part in parts]
                yield torch.tensor(x), torch.tensor([y])

            f.close()

        return iter(_epsilon_generator())

    def __len__(self):
        return self.length


class EpsilonLogisticModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.stack = torch.nn.Sequential(
            torch.nn.Linear(2000, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.stack(x)


def _epsilon_hint(location):
    print("-" * 80)
    print("\033[1;31mError: Data file not found\033[0m")
    print("Hint: Download the epsilon dataset (both files) from")
    print("  https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#epsilon")
    print("and extract it (using bunzip2) to this directory:")
    print("  " + str(location.parent))
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
