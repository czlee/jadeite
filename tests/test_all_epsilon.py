"""Simple regression tests that just run every listed experiment class once
each on a small epsilon dataset.

For this to work, the epsilon-small dataset needs to be set up.  This is just
a smaller version of the epsilon dataset.

The epsilon dataset can be found at:
  https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#epsilon

In particular:

  https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2
  https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2

Then use `head` to grab a few hundred lines from the full files. For example:

    wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2
    bunzip2 epsilon_normalized.t.bz2
    mv epsilon_normalized.t epsilon_normalized.t.full
    head epsilon_normalized.t.full -n 1000 > epsilon_normalized
    tail epsilon_normalized.t.full -n 200 > epsilon_normalized.t
"""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import argparse
import json
import logging
import shutil
import unittest
from pathlib import Path

import torch

import data.epsilon as epsilon
import metrics
from experiments import (DynamicPowerOverTheAirExperiment, FederatedAveragingExperiment,
                         OverTheAirExperiment, SimpleExperiment)


class TestAllExperimentsWithEpsilon(unittest.TestCase):

    def setUp(self):
        try:
            self.train_dataset = epsilon.EpsilonDataset(train=True, small=True, verbose=False)
            self.test_dataset = epsilon.EpsilonDataset(train=False, small=True, verbose=False)
        except FileNotFoundError:
            raise unittest.SkipTest

        self.results_dir = Path("/tmp/jadeite-test")
        self.results_dir.mkdir(exist_ok=True)

        self.file_handler = logging.FileHandler(self.results_dir / "output.log")
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(self.file_handler)

    def tearDown(self):
        self.file_handler.close()
        root_logger = logging.getLogger()
        root_logger.removeHandler(self.file_handler)
        shutil.rmtree(self.results_dir)

    @staticmethod
    def get_args(cls):
        """Returns an argparse.Namespace object suitable for passing to the
        constructor of `cls`."""
        parser = argparse.ArgumentParser()
        cls.add_arguments(parser)
        if 'clients' in cls.default_params:
            return parser.parse_args(["--clients", "4"])
        else:
            return parser.parse_args([])

    def assertFileProduced(self, path):  # noqa: N802
        assert path.exists(), f"Path does not exist: {path}"
        assert path.is_file(), f"Path is not a file: {path}"
        self.assertGreater(path.stat().st_size, 50)
        if path.suffix == '.json':
            # check that json file is well-formed
            with open(path, 'r') as f:
                json.load(f)

    def test_simple(self):
        model = epsilon.EpsilonLogisticModel()
        loss_fn = torch.nn.functional.binary_cross_entropy
        metric_fns = {"accuracy": metrics.binary_accuracy}
        args = self.get_args(SimpleExperiment)
        experiment = SimpleExperiment.from_arguments(
            self.train_dataset, self.test_dataset, model, loss_fn, metric_fns, self.results_dir, args)
        experiment.run()

        self.assertFileProduced(self.results_dir / "training.csv")
        self.assertFileProduced(self.results_dir / "evaluation.json")
        self.assertFileProduced(self.results_dir / "output.log")

    def _test_fedavg(self, experiment_class, send='deltas'):
        loss_fn = torch.nn.functional.binary_cross_entropy
        metric_fns = {"accuracy": metrics.binary_accuracy}
        args = self.get_args(experiment_class)
        args.send = send
        experiment = experiment_class.from_arguments(
            self.train_dataset, self.test_dataset, epsilon.EpsilonLogisticModel,
            loss_fn, metric_fns, self.results_dir, args)
        experiment.run()

        self.assertFileProduced(self.results_dir / "training.csv")
        self.assertFileProduced(self.results_dir / "evaluation.json")
        self.assertFileProduced(self.results_dir / "output.log")

    def test_fedavg_deltas(self):
        self._test_fedavg(FederatedAveragingExperiment, send='deltas')

    def test_fedavg_params(self):
        self._test_fedavg(FederatedAveragingExperiment, send='params')

    def test_overtheair_deltas(self):
        self._test_fedavg(OverTheAirExperiment, send='deltas')

    def test_overtheair_params(self):
        self._test_fedavg(OverTheAirExperiment, send='params')

    @unittest.skip("not yet implemented")
    def test_dynpower_deltas(self):
        self._test_fedavg(DynamicPowerOverTheAirExperiment, send='deltas')

    @unittest.skip("not yet implemented")
    def test_dynpower_params(self):
        self._test_fedavg(DynamicPowerOverTheAirExperiment, send='params')
