"""Classes for digital federated experiments.

Experiment classes are supposed to be, as far as possible, agnostic to models,
loss functions and optimizers. They take care of training, testing and logging.
"""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import logging
from math import log2
from typing import Sequence

import torch

from .federated import BaseFederatedExperiment

logger = logging.getLogger(__name__)


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
        'channel_uses': None,
    })

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # take note of number of model parameters
        flattened = self.flatten_state_dict(self.global_model.state_dict())
        self.nparams = flattened.numel()

        if self.params['channel_uses'] is None:
            # override with the number of parameters to estimate
            self.params['channel_uses'] = self.nparams
            logger.debug(f"Setting number of channel uses to: {self.params['channel_uses']}")

        logger.info(f"Number of bits available per channel use: {self.bits}")
        logger.info(f"Number of bits available in total: {self.bits * self.params['channel_uses']}")

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
        parser.add_argument("-s", "--channel-uses", type=float,
            help="Number of channel uses (default: same as number of model components)")

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

    def log_evaluation(self, evaluation_dict):
        evaluation_dict['bits_per_channel_use'] = self.bits
        evaluation_dict['bits_per_round'] = self.bits * self.params['channel_uses']
        return super().log_evaluation(evaluation_dict)


class SimpleStochasticQuantizationMixin:
    """Mixin for stochastic quantization functionality.

    This quantization uses evenly spaced bins within a symmetric quantization
    range [-M, M]. It has no adaptive intelligence. The parameter `M` must be
    specified as the `quantization_range` in the parameters. For values within
    [-M, M], quantization is either up or down randomly, so that it is equal in
    expectation to the true value. Values outside [-M, M] will just be pulled
    back to the range boundary, i.e., -M or M. See the docstring for the
    `quantize()` method for details.

    There are unit tests for this mixin, see tests/test_quantize.py or invoke
    from the repository root directory:

        python -m unittest tests.test_quantize

    """

    default_params_to_add = {
        'quantization_range': 1.0,
        'zero_bits_strategy': 'min-one',
    }

    @classmethod
    def add_arguments(cls, parser):
        """Adds relevant command-line arguments to the given `parser`, which
        should be an `argparse.ArgumentParser` object.
        """
        parser.add_argument("-M", "--quantization-range", type=float,
            help="Quantization range, [-M, M]")
        parser.add_argument("--zero-bits-strategy", choices=['min-one', 'read-zero'],
            help="What to do if there aren't enough bits (min-one = require at "
                 "least one bit per parameter, even if it violates the power "
                 "constraint; read-zero = interpret parameters without bits as "
                 "zero)")
        super().add_arguments(parser)

    def _binwidths(self, nbits: torch.Tensor) -> torch.Tensor:
        """Convenience function to compute bin widths relating to `nbits`.
        Elements corresponding to `nbits == 0` are returned as NaN, unless
        `self.params['zero_bits_strategy'] == 'min_one'`, in which case they are
        treated as if `nbits` were 1.

        `nbits` doesn't have to be of an integer dtype, but it does need to
        contain only integers."""
        nbits = nbits.type(torch.float64)
        assert torch.equal(nbits, nbits.floor())

        M = self.params['quantization_range']  # noqa: N806
        if self.params['zero_bits_strategy'] == 'min-one':
            nbits = torch.maximum(nbits, torch.ones_like(nbits))
        elif self.params['zero_bits_strategy'] == 'read-zero':
            nbits[nbits == 0] = float('nan')

        nbins = 2 ** nbits - 1
        binwidths = 2 * M / nbins
        return binwidths

    def quantize(self, values: torch.Tensor, nbits: torch.Tensor) -> torch.Tensor:
        """Quantizes the given `values` to the corresponding number of bits in
        `nbits`. The two arrays passed in must be the same size.

        To quantize, the range [-M, M], where `M` is the `quantization_range`
        parameter, is divided equally into `2 ** nbits - 1` bins. For example,
        if M = 5:

        nbits  value   returns indices           bin values
          1      0     0 w.p. 0.5, 1 w.p. 0.5    0 means -5,    1 means 5
          1      3     0 w.p. 0.2, 1 w.p. 0.8    0 means -5,    1 means 5
          1      8     1 w.p. 1                  1 means 5
          2      0     1 w.p. 0.5, 2 w.p. 0.5    1 means -5/3,  2 means 5/3
          2      4     2 w.p. 0.3, 3 w.p. 0.7    2 means  5/3,  3 means 5
          3     -1     2 w.p. 0.2, 3 w.p. 0.8    2 means -15/7, 3 means -5/7

        If nbits is 0, the corresponding index returned is always 0, and should
        be interpreted to mean 0.
        """
        assert values.shape == nbits.shape, f"shape mismatch: {nbits.shape} vs {nbits.shape}"
        M = self.params['quantization_range']  # noqa: N806

        binwidths = self._binwidths(nbits)
        clipped = values.clip(-M, M)
        scaled = (clipped + M) / binwidths    # scaled to 0:nbins
        lower = torch.floor(scaled)           # rounded down
        remainder = scaled - lower            # probability we want to round up
        assert remainder.ge(0).all() and remainder.le(1).all()
        round_up = torch.rand_like(remainder) < remainder
        indices = (lower + round_up).type(torch.int64)
        indices[binwidths.isnan()] = 0      # override special case
        return indices

    def unquantize(self, indices: torch.Tensor, nbits: torch.Tensor) -> torch.Tensor:
        """Returns the scaled quantized values corresponding to the given
        indices. For example, if M = 5:

        nbits  index  returns value
          0      0          0
          1      0         -5
          1      1          5
          2      2          5/3
          3      1         -25/7

        (The method name is a bit of a misnomer. You obviously can't
        "unquantize" or undo a reduction in information. But the point is that
        it transforms the indices back to a usable space.)
        """
        assert indices.shape == nbits.shape, f"shape mismatch: {indices.shape} vs {nbits.shape}"
        M = self.params['quantization_range']  # noqa: N806
        binwidths = self._binwidths(nbits)
        values = indices * binwidths - M
        values[binwidths.isnan()] = 0        # override special case
        return values


class SimpleQuantizationFederatedExperiment(
        SimpleStochasticQuantizationMixin,
        BaseDigitalFederatedExperiment):
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

    default_params = BaseDigitalFederatedExperiment.default_params.copy()
    default_params.update(SimpleStochasticQuantizationMixin.default_params_to_add)

    def run(self):
        self.cursor = 0
        super().run()

    def transmit_and_aggregate(self, records: dict):
        self.bits_per_channel = self.advance_bits_per_channel()
        super().transmit_and_aggregate(records)

    def advance_bits_per_channel(self):
        """Returns a list of the number of bits each model parameter (in the
        state dict) should use. This evenly divides the total number of bits
        available (being `self.bits * self.params['channel_uses']`) among the
        number of model parameters (i.e., numbers to be sent). The leftover bits
        are allocated on a rotating basis, tracked by `self.cursor`.
        """
        s = self.params['channel_uses']
        d = self.nparams
        total_bits = int(self.bits * s)
        lengths = torch.ones((1, d), dtype=torch.int64, device=self.device) * (total_bits // d)
        nspare = total_bits - lengths.sum()
        extra_locations = torch.arange(self.cursor, self.cursor + nspare, device=self.device) % d
        lengths[0, extra_locations] += 1
        assert lengths.sum() == total_bits
        self.cursor = (self.cursor + nspare) % d
        return lengths

    def client_transmit(self, model: torch.nn.Module) -> torch.Tensor:
        values = self.get_values_to_send(model)
        lengths = self.bits_per_channel
        assert values.shape == lengths.shape, f"shape mismatch: {values.shape} vs {lengths.shape}"
        indices = self.quantize(values, lengths)
        return indices

    def server_receive(self, transmissions):
        lengths = self.bits_per_channel
        unquantized = [self.unquantize(indices, lengths) for indices in transmissions]
        client_average = torch.stack(unquantized, 0).mean(0)
        self.update_global_model(client_average)
