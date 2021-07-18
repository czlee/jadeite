"""Some simple unit tests for certain things in the federated experiment
classes."""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import shutil
import unittest
from pathlib import Path

import torch

import data.epsilon as epsilon
from experiments.federated import BaseFederatedExperiment


class TestFederatedExperiment(unittest.TestCase):

    def setUp(self):
        train_dataset = epsilon.EpsilonDataset(train=True, small=True, verbose=False)
        test_dataset = epsilon.EpsilonDataset(train=False, small=True, verbose=False)
        global_model = epsilon.EpsilonLogisticModel()
        self.client_model = epsilon.EpsilonLogisticModel()
        client_optimizer = torch.optim.SGD(self.client_model.parameters(), lr=1e-2)
        loss_fn = torch.nn.functional.binary_cross_entropy
        self.results_dir = Path("/tmp/jadeite-test")
        self.results_dir.mkdir(exist_ok=True)

        self.experiment = BaseFederatedExperiment(
            [train_dataset], test_dataset, [self.client_model], global_model,
            loss_fn, {}, [client_optimizer], self.results_dir)

    def tearDown(self):
        shutil.rmtree(self.results_dir)

    def assertStateDictEqual(self, sd1, sd2):  # noqa: N802
        self.assertEqual(sd1.keys(), sd2.keys())
        for key in sd1.keys():
            self.assertTrue(torch.equal(sd1[key], sd2[key]))

    def test_flattening(self):
        state_dict = self.client_model.state_dict()
        flattened = self.experiment.flatten_state_dict(state_dict)
        new_state_dict = self.experiment.unflatten_state_dict(flattened)
        self.assertStateDictEqual(state_dict, new_state_dict)
