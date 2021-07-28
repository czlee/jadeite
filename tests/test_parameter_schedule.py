"""Simple unit tests for bits per tx parameter partition."""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import unittest

import torch

from experiments.digital import BaseDigitalFederatedExperiment


class ExperimentTest(BaseDigitalFederatedExperiment):
    """Overrides the `bits` property to allow it to be set easily."""

    def __init__(self):
        # Override to remove the need to initialize with datasets etc.
        # Set up only what attributes are necessary for this to work.
        self.params = {}
        self.device = 'cpu'

    @property
    def bits(self):
        return self._bits

    @bits.setter
    def bits(self, value):
        self._bits = value


class TestParameterScheduling(unittest.TestCase):

    def setUp(self):
        self.experiment = ExperimentTest()

    def calculate_bits_per_tx_parameter(self, s, d, bpcu, r, i):
        self.experiment.nparams = d
        self.experiment.params['channel_uses'] = s
        self.experiment.current_round = r
        self.experiment.bits = bpcu
        return self.experiment.bits_per_tx_parameter(i)

    def test_bits_per_tx_parameter_aligned(self):
        self.experiment.params['parameter_schedule'] = 'aligned'

        specs = [
            (dict(s=10, d=10, bpcu=0.5, r=0, i=0),
                torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])),

            (dict(s=10, d=10, bpcu=0.5, r=3, i=1),
                torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])),

            (dict(s=10, d=10, bpcu=0.5, r=4, i=1),
                torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])),

            (dict(s=10, d=10, bpcu=0.3, r=0, i=0),
                torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])),

            (dict(s=10, d=10, bpcu=0.3, r=0, i=1),
                torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])),

            (dict(s=10, d=10, bpcu=0.3, r=0, i=2),
                torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])),

            (dict(s=10, d=10, bpcu=0.3, r=1, i=0),
                torch.tensor([[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]])),

            (dict(s=10, d=10, bpcu=0.3, r=1, i=2),
                torch.tensor([[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]])),

            (dict(s=10, d=10, bpcu=0.3, r=100, i=17),
                torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])),

            (dict(s=10, d=10, bpcu=3, r=7, i=5),
                torch.tensor([[3, 3, 3, 3, 3, 3, 3, 3, 3, 3]])),

            (dict(s=4, d=10, bpcu=3, r=0, i=0),
                torch.tensor([[2, 2, 1, 1, 1, 1, 1, 1, 1, 1]])),

            (dict(s=4, d=10, bpcu=3, r=5, i=4),
                torch.tensor([[2, 2, 1, 1, 1, 1, 1, 1, 1, 1]])),
        ]

        for kwargs, expected in specs:
            calculated = self.calculate_bits_per_tx_parameter(**kwargs)
            torch.testing.assert_equal(calculated, expected)

    def test_bits_per_tx_parameter_staggered(self):
        self.experiment.params['parameter_schedule'] = 'staggered'

        specs = [
            (dict(s=10, d=10, bpcu=0.5, r=0, i=0),
                torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])),

            (dict(s=10, d=10, bpcu=0.5, r=3, i=1),
                torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])),

            (dict(s=10, d=10, bpcu=0.5, r=4, i=1),
                torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])),

            (dict(s=10, d=10, bpcu=0.3, r=0, i=0),
                torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])),

            (dict(s=10, d=10, bpcu=0.3, r=0, i=1),
                torch.tensor([[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]])),

            (dict(s=10, d=10, bpcu=0.3, r=0, i=2),
                torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 0]])),

            (dict(s=10, d=10, bpcu=0.3, r=1, i=0),
                torch.tensor([[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]])),

            (dict(s=10, d=10, bpcu=0.3, r=1, i=2),
                torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 1]])),

            (dict(s=10, d=10, bpcu=0.3, r=100, i=17),
                torch.tensor([[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]])),

            (dict(s=10, d=10, bpcu=3, r=7, i=5),
                torch.tensor([[3, 3, 3, 3, 3, 3, 3, 3, 3, 3]])),

            (dict(s=4, d=10, bpcu=3, r=0, i=0),
                torch.tensor([[2, 2, 1, 1, 1, 1, 1, 1, 1, 1]])),

            (dict(s=4, d=10, bpcu=3, r=5, i=4),
                torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 2, 2]])),
        ]

        for kwargs, expected in specs:
            calculated = self.calculate_bits_per_tx_parameter(**kwargs)
            torch.testing.assert_equal(calculated, expected)
