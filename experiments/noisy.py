"""Variants that don't involve communication constraints in the proper sense,
but do add artificial noise to models during training.
"""

# Chuan-Zheng Lee <czlee@stanford.edu>
# August 2021

import logging
from math import sqrt

import torch

from .experiment import SimpleExperiment
from .federated import BaseFederatedExperiment

logger = logging.getLogger(__name__)


class SimpleWithNoiseExperiment(SimpleExperiment):
    """Adds Gaussian noise to the model before every training round."""

    default_params = SimpleExperiment.default_params.copy()
    default_params.update({
        'noise': 1.0,
    })

    description = """\
        Non-federated machine learning, but with Gaussian noise added to the
        model just before every training round.
    """

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


class FederatedAveragingWithNoiseExperiment(BaseFederatedExperiment):
    """Adds Gaussian noise to the model just before every global model
    synchronization.

    This differs from the analog scheme in that this has no power constraint,
    though it should in principle be equivalent to the analog scheme with fixed
    parameter radius (with noise level adjusted).
    """

    default_params = BaseFederatedExperiment.default_params.copy()
    default_params.update({
        'noise': 1.0,
    })

    description = """\
        Federated averaging, but with Gaussian noise added to the
        model just before every global model synchronization.
    """

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("-N", "--noise", type=float,
            help="Noise level (variance), σₙ²")
        super().add_arguments(parser)

    def transmit_and_aggregate(self):
        client_values = [self.get_values_to_send(model) for model in self.client_models]
        client_average = torch.stack(client_values, 0).mean(0)

        σₙ = sqrt(self.params['noise'])  # stdev
        noise = torch.normal(0.0, σₙ, size=client_average.size()).to(self.device)
        noisy_average = client_average + noise

        self.update_global_model(noisy_average)
