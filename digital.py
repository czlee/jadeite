"""Classes for digital federated experiments.

Experiment classes are supposed to be, as far as possible, agnostic to models,
loss functions and optimizers. They take care of training, testing and logging.
"""

from math import log2
from typing import Sequence

import torch

from federated import BaseFederatedExperiment


class BaseDigitalFederatedExperiment(BaseFederatedExperiment):
    """Base class for digital federated experiments.

    By "digital federated experiment", we mean, "federated experiment where
    clients can send bits at up to the Shannon capacity of the channel, equally
    divided among clients". This is given by equation (13) in our GLOBECOM 2020
    paper,
                     s            n P
                k = --- log ( 1 + --- )
                    2 n    2       σₙ²
    """

    default_params = BaseFederatedExperiment.default_params.copy()
    default_params.update({
        'noise': 1.0,
        'power': 1.0,
    })

    @property
    def bits(self):
        """Bits per channel use."""
        n = self.params['clients']
        P = self.params['power']    # noqa: N806
        σₙ2 = self.params['noise']
        return log2(1 + n * P / σₙ2) / (2 * n)

    @classmethod
    def add_arguments(cls, parser):
        """Adds relevant command-line arguments to the given `parser`, which
        should be an `argparse.ArgumentParser` object.
        """
        parser.add_argument("-N", "--noise", type=float,
            help="Noise level (variance), σₙ²")
        parser.add_argument("-P", "--power", type=float,
            help="Power level, P")

        super().add_arguments(parser)

    def client_transmit(self, model) -> torch.Tensor:
        """Should return a tensor containing the number of bits specified by
        `self.bits`.  Must be implemented by subclasses."""
        raise NotImplementedError

    def server_receive(self, transmissions: Sequence[torch.Tensor]):
        """Should update the global `model` given the `transmissions` received
        (errorlessly) from the channel (assumed to be using reliable coding).
        Must be implemented by subclasses."""
        raise NotImplementedError

    def transmit_and_aggregate(self, records: dict):
        """Transmits model data over the channel as bits, receives the bits
        errorlessly at the server and updates the model at the server"""
        transmissions = [self.client_transmit(model) for model in self.client_models]
        self.server_receive(transmissions)
