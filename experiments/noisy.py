"""Variants that don't involve communication constraints in the proper sense,
but do add artificial noise to models during training.
"""

# Chuan-Zheng Lee <czlee@stanford.edu>
# August 2021

import logging
from math import sqrt

import torch

from .experiment import SimpleExperiment
from .federated import FederatedAveragingExperiment

logger = logging.getLogger(__name__)


class AddNoiseToModelMixin:
    """Mixin to add noise directly to a model."""

    default_params_to_add = {
        'noise': 1.0,
    }

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("-N", "--noise", type=float,
            help="Noise level (variance), σₙ²")
        super().add_arguments(parser)

    def add_noise_to_model(self, model):
        new_state_dict = {}

        for key, value in model.state_dict().items():
            σₙ = sqrt(self.params['noise'])  # stdev
            noise = torch.normal(0.0, σₙ, size=value.size()).to(self.device)
            new_state_dict[key] = value + noise

        model.load_state_dict(new_state_dict)


class SimpleWithNoiseExperiment(AddNoiseToModelMixin, SimpleExperiment):
    """Adds Gaussian noise to the model before every training round.

    For convenience, noise is added during the testing phase, so the model is
    tested both before and after adding noise.
    """

    default_params = SimpleExperiment.default_params.copy()
    default_params.update(AddNoiseToModelMixin.default_params_to_add)

    description = """\
        Non-federated machine learning, but with Gaussian noise added to the
        model once per round.
    """

    def test(self):
        prenoise_results = self._test(self.test_dataloader, self.model, record_prefix='prenoise_')
        self.add_noise_to_model(self.model)
        postnoise_results = self._test(self.test_dataloader, self.model, record_prefix='postnoise_')
        return {**prenoise_results, **postnoise_results}


class FederatedAveragingWithNoiseExperiment(AddNoiseToModelMixin, FederatedAveragingExperiment):
    """Adds Gaussian noise to the model just before every global model
    synchronization.

    This differs substantially in concept from the analog scheme in
    `experiments.analog.OverTheAirExperiment`, though it should end up being
    equivalent in principle. Differences:

     - There is no power constraint or any sort of power scaling.
     - Noise is not added as part of transmission, but is artificially added
       only after the averaging step.

    This differs from the analog scheme in that this has no power constraint,
    though it should in principle be equivalent to the analog scheme with fixed
    parameter radius (with noise level adjusted).

    For convenience, noise is added during the testing phase, so the model is
    tested both before and after adding noise.
    """

    default_params = FederatedAveragingExperiment.default_params.copy()
    default_params.update(AddNoiseToModelMixin.default_params_to_add)

    description = """\
        Federated averaging, but with Gaussian noise added to the
        model just after every averaging.
    """

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("-N", "--noise", type=float,
            help="Noise level (variance), σₙ²")
        super().add_arguments(parser)

    def test(self):
        prenoise_results = self._test(self.test_dataloader, self.global_model,
                                      record_prefix='prenoise_')

        # Since we're adding noise after the averaging step, we need to keep
        # client models in sync "manually" after we add noise.
        self.add_noise_to_model(self.global_model)
        for model in self.client_models:
            model.load_state_dict(self.global_model.state_dict())

        postnoise_results = self._test(self.test_dataloader, self.global_model,
                                       record_prefix='postnoise_')
        return {**prenoise_results, **postnoise_results}
