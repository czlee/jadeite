"""Classes for digital federated experiments.

Experiment classes are supposed to be, as far as possible, agnostic to models,
loss functions and optimizers. They take care of training, testing and logging.
"""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

from math import log2
from typing import Sequence

import torch

from .federated import BaseFederatedExperiment


class BaseDigitalFederatedExperiment(BaseFederatedExperiment):
    """Base class for digital federated experiments.

    By "digital federated experiment", we mean, "federated experiment where
    clients can send bits at up to the Shannon capacity of the channel, equally
    divided among clients". This is given by equation (13) in our GLOBECOM 2020
    paper,
                     s            n P
                k = --- log₂( 1 + --- )
                    2 n            σₙ²
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

    def client_transmit(self, model: torch.nn.Module) -> torch.Tensor:
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
        errorlessly at the server and updates the model at the server."""
        transmissions = [self.client_transmit(model) for model in self.client_models]
        self.server_receive(transmissions)


class SimpleQuantizationFederatedExperiment(BaseDigitalFederatedExperiment):
    """Digital federated experiment that quantizes each component of the model
    to the number of bits available for that component. Bits are allocated using
    the following simple (and presumably inefficient) strategy: If the number of
    bits isn't a multiple of the number of components (this will typically be
    the case), the leftover bits are rotated between components. For example, if
    the model has 10 components and there are 13 bits available, then:

     - in round 1, components 0--2 get 2 bits, components 3--9 get 1 bit.
     - in round 2, components 3--5 get 2 bits, and the rest get 1 bit.
     - in round 3, components 6--8 get 2 bits, and the rest get 1 bit.
     - in round 4, components 9, 0 and 1 get 2 bits, and the rest get one bit.

    and so on. If there are fewer bits available than components, then all bits
    are rotated between components in the same way.

    The number of channel uses is assumed to be equal to the number of
    components. (This will probably change in a near-future implementation.)"""

    def transmit_and_aggregate(self, records: dict):
        self.cursor = 0
        super().transmit_and_aggregate(records)

    def divide_bits_per_channel(self, ncomponents):
        total_bits = self.bits * ncomponents
        lengths = [total_bits // ncomponents] * ncomponents
        nspare = total_bits - sum(lengths)
        for i in range(self.cursor, self.cursor + nspare):
            lengths[i % ncomponents] += 1
        self.cursor = (self.cursor + nspare) % ncomponents
        return lengths

    def client_transmit(self, model: torch.nn.Module) -> torch.Tensor:
        values = self.flatten_state_dict(model)
        lengths = self.divide_bits_per_channel(values.numel())
        assert len(values) == len(lengths)

    # things that need to happen
    #  - how do we quantize? like what maximum/range do we assume?
    #    do we randomize rounding direction?
    #  - probably best to add option to send deltas rather than quantities
    #  - scale step size to utilize all power
