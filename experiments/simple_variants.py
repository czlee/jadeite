"""Classes for variants on non-federated learning.
"""

import logging
from math import sqrt

import torch

from .experiment import SimpleExperiment

logger = logging.getLogger(__name__)


class SimpleExperimentWithNoise(SimpleExperiment):
    """Adds Gaussian noise to the model before every training round."""

    default_params = SimpleExperiment.default_params.copy()
    default_params.update({
        'noise': 1.0,
    })

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("-N", "--noise", type=float,
            help="Noise level (variance), σₙ²")
        super().add_arguments(parser)

    def train(self):
        self.add_noise_to_model(self.model)
        return super().train()

    def add_noise_to_model(self, model):
        new_state_dict = {}

        for key, value in model.state_dict().items():
            σₙ = sqrt(self.params['noise'])  # stdev
            noise = torch.normal(0.0, σₙ, size=value.size()).to(self.device)
            new_state_dict[key] = value + noise

        model.load_state_dict(new_state_dict)
