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

from .analog import ExponentialMovingAverageMixin
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

    This base class tracks the number of bits that can be used to represent each
    model parameter.  Subclasses can (and should) access this in
    `self.bits_per_tx_parameter`, normally from their implementation of
    `self.client_transmit()`.
    """

    default_params = BaseFederatedExperiment.default_params.copy()
    default_params.update({
        'noise': 1.0,
        'power': 1.0,
        'channel_uses': None,
        'parameter_schedule': 'staggered',
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
        digital_args = parser.add_argument_group(title="Digital federated parameters")
        digital_args.add_argument("-N", "--noise", type=float,
            help="Noise level (variance), σₙ²")
        digital_args.add_argument("-P", "--power", type=float,
            help="Power level, P")
        digital_args.add_argument("-s", "--channel-uses", type=float,
            help="Number of channel uses (default: same as number of model components)")
        digital_args.add_argument("-ps", "--parameter-schedule", choices=['aligned', 'staggered'],
            help="Parameter scheduling system. In general, bits are split evenly among parameters, "
                 "and there will be some left over due to integer division. This option describes "
                 "how these leftover bits are rotated among parameters. In 'aligned', all clients "
                 "assign the leftover bits to the same parameters. If 'staggered', client 2 "
                 "assigns them to start just after the parameters that client 1 assigned, and so "
                 "on. In precise terms, with k leftover bits, in round r, with all expressions "
                 "taken modulo d (number of parameters): aligned = [r*k, ..., r*(k+1)-1], "
                 "staggered = [(r+i)*k, ..., (r+i)*(k+1)-1].")

        super().add_arguments(parser)

    def run(self):
        self._bits_cursor = 0
        super().run()

    def bits_per_tx_parameter(self, client: int):
        """Returns a list of the number of bits each model parameter (in the
        state dict) should use. This evenly divides the total number of bits
        available (being `self.bits * self.params['channel_uses']`) among the
        number of model parameters (i.e., numbers to be sent).
        """
        d = self.nparams
        total_bits = int(self.bits * self.params['channel_uses'])
        lengths = torch.full((1, d), total_bits // d, dtype=torch.int64, device=self.device)
        nspare = total_bits - lengths.sum()

        # Proactively take numbers mod d often, to avoid potential integer
        # overflow issues at high round numbers. (This is probably overly
        # conservative, but better safe than sorry.) Laws of modulo arithmetic
        # say that if a ≡ b and c ≡ d then ac ≡ bd, so this should be okay, as
        # in, if we only took modulo d at the very end, it should be the same.
        r = self.current_round % d
        if self.params['parameter_schedule'] == 'staggered':
            r = (r + client) % d
        start = (r * nspare) % d

        extras_pos = torch.arange(start, start + nspare, device=self.device) % d
        lengths[0, extras_pos] += 1
        assert lengths.sum() == total_bits, repr(lengths)
        assert lengths.max() - lengths.min() <= 1, repr(lengths)
        return lengths

    def client_transmit(self, client: int, model: torch.nn.Module) -> torch.Tensor:
        """Returns a row tensor containing the number of bits specified by
        `self.bits`, representing the bits to be transmitted from the client
        with index `index`. Subclasses must implement this method.

        Subclasses might find `self.bits_per_tx_parameter` useful in their
        implementation of this method.
        """
        raise NotImplementedError

    def server_receive(self, transmissions: Sequence[torch.Tensor]):
        """Updates the global `model` given the `transmissions` received
        (errorlessly) from the channel (assumed to be using reliable coding).
        Subclasses must implement this method.
        """
        raise NotImplementedError

    def transmit_and_aggregate(self):
        """Transmits model data over the channel as bits, receives the bits
        errorlessly at the server and updates the model at the server.
        """
        transmissions = [self.client_transmit(i, model) for i, model in enumerate(self.client_models)]
        self.server_receive(transmissions)

    def log_evaluation(self, evaluation_dict):
        evaluation_dict['bits_per_channel_use'] = self.bits
        evaluation_dict['bits_per_round'] = self.bits * self.params['channel_uses']
        return super().log_evaluation(evaluation_dict)


class QuantizationWithEqualBinsMixin:
    """Mixin for simple quantization functionality.

    This quantization uses evenly spaced bins within a symmetric quantization
    range [-qrange, qrange]. The parameter `qrange` must be passed in on every
    call, and it is for the caller to ensure that the values provided are
    consistent.

    If the `rounding_method` parameter is `"stochastic"`, then for values within
    [-qrange, qrange], quantization is either up or down randomly, so that it is
    equal in expectation to the true value. Values outside [-qrange, qrange]
    will just be pulled back to the range boundary, i.e., `-qrange` or `qrange`.
    See the docstring for the `quantize_stochastic()` method for details.

    If the `rounding_method` parameter is `"deterministic"`, then values are
    simply rounded to the nearest quantization level.

    There are unit tests for this mixin, see tests/test_quantize.py or invoke
    from the repository root directory:

        python -m unittest tests.test_quantize

    (The reason this is a mixin is that I could imagine wanting to add other
    types of quantization, so this would make it easier to swap quantization
    strategies without affecting other parts of a digital scheme.)
    """

    default_params_to_add = {
        'zero_bits_strategy': 'read-zero',
        'rounding_method': 'stochastic',
    }

    @classmethod
    def add_arguments(cls, parser):
        quant_args = parser.add_argument_group(title="Quantization options")
        quant_args.add_argument("-zbs", "--zero-bits-strategy",
            choices=['min-one', 'read-zero', 'exclude'],
            help="What to do if there aren't enough bits. min-one = require at "
                 "least one bit per parameter, even if it violates the power "
                 "constraint. read-zero = interpret parameters without bits as "
                 "zero. exclude = exclude that client's parameter when averaging.")
        quant_args.add_argument("-rdm", "--rounding-method", choices=['stochastic', 'deterministic'],
            help="Rounding method. stochastic = round up or down randomly so "
                 "that the rounded value is equal to the true value in "
                 "expectation. deterministic = round to the nearest "
                 "quantization level, with round-to-even tiebreak.")

        super().add_arguments(parser)

    def get_binwidths(self, nbits: torch.Tensor, qrange: float) -> torch.Tensor:
        """Convenience function to compute bin widths relating to `nbits`.
        Elements corresponding to `nbits == 0` are returned as NaN, unless
        `self.params['zero_bits_strategy'] == 'min_one'`, in which case they are
        treated as if `nbits` were 1.

        `nbits` doesn't have to be of an integer dtype, but it does need to
        contain only integers.
        """
        nbits = nbits.type(torch.float64)
        assert torch.equal(nbits, nbits.floor())
        assert torch.lt(nbits, 64).all(), "can't quantize to more than 63 bits!"

        if self.params['zero_bits_strategy'] == 'min-one':
            nbits = torch.maximum(nbits, torch.ones_like(nbits))
        elif self.params['zero_bits_strategy'] in ['read-zero', 'exclude']:
            nbits[nbits == 0] = float('nan')

        nbins = 2 ** nbits - 1
        binwidths = 2 * qrange / nbins
        return binwidths

    def quantize(self, values: torch.Tensor, nbits: torch.Tensor, qrange: float) -> torch.Tensor:
        """Quantizes the given `values` to the corresponding number of bits in
        `nbits`. The two arrays passed in must be the same size.

        To quantize, the range [-qrange, qrange], is divided equally into `2 **
        nbits - 1` bins. Values are then rounded to a nearby quantization level.
        The rounding is either done deterministically to the nearest level, or
        stochastically to maintain equality in expectation, depending on how the
        `rounding_method` parameter is set. See the docstring for
        `quantize_deterministic()` and `quantize_stochastic()` for examples.

        If nbits is 0, the corresponding index returned is always 0, and should
        be interpreted to mean 0.
        """
        assert values.shape == nbits.shape, f"shape mismatch: {nbits.shape} vs {nbits.shape}"

        if self.params['rounding_method'] == 'stochastic':
            return self.quantize_stochastic(values, nbits, qrange)
        if self.params['rounding_method'] == 'deterministic':
            return self.quantize_deterministic(values, nbits, qrange)

    def quantize_deterministic(self, values: torch.Tensor, nbits: torch.Tensor,
                               qrange: float) -> torch.Tensor:
        """Quantizes deterministically by rounding to the nearest quantization
        level. For example, if qrange = 5:

        nbits  value   returns indices       unquantized value
          1      0        0 (even tiebreak)      -5
          1      3        1                       5
          1      8        1                       5
          2      0        2 (even tiebreak)       5/3
          2      3        2                       5/3
          3     -1        3                      -5/7
        """
        binwidths = self.get_binwidths(nbits, qrange)
        clipped = values.clip(-qrange, qrange)
        scaled = (clipped + qrange) / binwidths  # scaled to 0:nbins
        indices = torch.round(scaled).type(torch.int64)
        indices[binwidths.isnan()] = 0           # override special case
        return indices

    def quantize_stochastic(self, values: torch.Tensor, nbits: torch.Tensor,
                            qrange: float) -> torch.Tensor:
        """Quantizes stochastically to maintain equality in expectation. For
        example, if qrange = 5:

        nbits  value   returns indices           bin values
          1      0     0 w.p. 0.5, 1 w.p. 0.5    0 means -5,    1 means 5
          1      3     0 w.p. 0.2, 1 w.p. 0.8    0 means -5,    1 means 5
          1      8     1 w.p. 1                  1 means 5
          2      0     1 w.p. 0.5, 2 w.p. 0.5    1 means -5/3,  2 means 5/3
          2      3     2 w.p. 0.6, 3 w.p. 0.4    2 means  5/3,  3 means 5
          3     -1     2 w.p. 0.2, 3 w.p. 0.8    2 means -15/7, 3 means -5/7
        """
        binwidths = self.get_binwidths(nbits, qrange)
        clipped = values.clip(-qrange, qrange)
        scaled = (clipped + qrange) / binwidths  # scaled to 0:nbins
        lower = torch.floor(scaled)              # rounded down
        remainder = scaled - lower               # probability we want to round up

        assert torch.logical_or(torch.logical_and(remainder.ge(0), remainder.le(1)),
                                remainder.isnan()).all(), remainder

        round_up = torch.rand_like(remainder) < remainder
        indices = (lower + round_up).type(torch.int64)
        indices[binwidths.isnan()] = 0           # override special case
        return indices

    def unquantize(self, indices: torch.Tensor, nbits: torch.Tensor, qrange: float) -> torch.Tensor:
        """Returns the scaled quantized values corresponding to the given
        indices. For example, if qrange = 5:

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
        binwidths = self.get_binwidths(nbits, qrange)
        values = indices * binwidths - qrange

        if self.params['zero_bits_strategy'] != 'exclude':
            values[binwidths.isnan()] = 0        # override special case

        return values

    def compute_client_average(self, all_unquantized):
        if self.params['zero_bits_strategy'] == 'exclude':
            # This replicates the np.nanmean() function (without using numpy), and can be
            # replaced when https://github.com/pytorch/pytorch/issues/21987 is closed.
            all_clients = torch.stack(all_unquantized, 0)
            nonnans_per_column = (~all_clients.isnan()).sum(0)
            client_average = all_clients.nansum(dim=0) / nonnans_per_column
            client_average = client_average.nan_to_num(0)
            return client_average

        else:
            return torch.stack(all_unquantized, 0).mean(0)


class SimpleQuantizationFederatedExperiment(
        QuantizationWithEqualBinsMixin,
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
    default_params.update(QuantizationWithEqualBinsMixin.default_params_to_add)
    default_params.update({
        'quantization_range': 1.0,
    })

    @classmethod
    def add_arguments(cls, parser):
        simple_quant_args = parser.add_argument_group(title="Simple quantization parameters")
        simple_quant_args.add_argument("-Q", "--quantization-range", type=float,
            help="Quantization range, [-Q, Q]")
        super().add_arguments(parser)

    def client_transmit(self, client: int, model: torch.nn.Module) -> torch.Tensor:
        values = self.get_values_to_send(model)
        lengths = self.bits_per_tx_parameter(client)
        assert values.shape == lengths.shape, f"shape mismatch: {values.shape} vs {lengths.shape}"
        qrange = self.params['quantization_range']
        indices = self.quantize(values, lengths, qrange)
        return indices

    def server_receive(self, transmissions):
        qrange = self.params['quantization_range']

        all_unquantized = []
        for i, indices in enumerate(transmissions):
            lengths = self.bits_per_tx_parameter(i)
            unquantized = self.unquantize(indices, lengths, qrange)
            all_unquantized.append(unquantized)

        client_average = self.compute_client_average(all_unquantized)
        self.update_global_model(client_average)


class DynamicRangeMixin(ExponentialMovingAverageMixin):
    """Provides functionality for dynamic range adjustment according to the
    following (very simple, presumptuous) protocol:

    Each client tracks an exponential moving average of some percentile (say,
    the 90th) among the parameters it transmits. Every few (say, 5) periods,
    clients "send" their current moving average values to the server, which
    itself takes some percentile rank among the clients (say, the 90th
    percentile), and sends that value back to the clients to be used by all
    clients as its quantization range.

    The dynamic quantization range is governed by four parameters:
    - The exponential moving average coefficient is set by the `ema_coefficient`
      parameter.
    - The update frequency is set by the `power_update_period` parameter.
    - The percentile rank is set by the `power_quantile` parameter (and is
      actually between 0 and 1).
    """
    default_params_to_add = ExponentialMovingAverageMixin.default_params_to_add.copy()
    default_params_to_add.update({
        'qrange_update_period': 1,
        'qrange_param_quantile': 1.0,
        'qrange_client_quantile': 1.0,
    })

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check parameter values for sensibility
        if not (0.0 <= self.params['qrange_param_quantile'] <= 1.0):
            logger.error("Dynamic quantization range parameter quantile must be between 0 and 1, found: "
                         f"{self.params['qrange_param_quantile']}")
        if not (0.0 <= self.params['qrange_client_quantile'] <= 1.0):
            logger.error("Dynamic quantization range client quantile must be between 0 and 1, found: "
                         f"{self.params['qrange_client_quantile']}")

        self.current_qrange = 1.0

    @classmethod
    def add_arguments(cls, parser):
        qrange_args = parser.add_argument_group(title="Dynamic quantization range parameters")
        qrange_args.add_argument("-qup", "--qrange-update-period", type=int, metavar="PERIOD",
            help="Number of rounds between quantization range updates")
        qrange_args.add_argument("-qpq", "--qrange-param-quantile", type=float, metavar="QUANTILE",
            help="Quantile among parameters to take to determine quantization range, between 0 and 1 "
                 "(1 means take the maximum parameter; this is done at each client)")
        qrange_args.add_argument("-qcq", "--qrange-client-quantile", type=float, metavar="QUANTILE",
            help="Quantile among clients to take to determine quantization range, between 0 and 1 "
                 "(1 means take the maximum among clients)")

        super().add_arguments(parser)

    def update_qrange_buffer(self, client: int, values: torch.Tensor):
        """Adds the current values to the dynamic range buffer. This should be
        called by subclasses in their implementation of client transmission,
        using the `values` returned by `get_values_to_send()`."""
        q = self.params['qrange_param_quantile']
        value_at_quantile = torch.quantile(values, q).item()
        self.records['param_quantile'] = value_at_quantile
        self.update_ema_buffer(client, value_at_quantile)

    def update_qrange(self):
        """Updates the current dynamic range. This should be called by
        subclasses in their implementation of server aggregation.
        """
        if self.current_round % self.params['qrange_update_period'] == 0:
            # update the current quantization range by looking at all the clients
            # (i.e. the clients "sent this information through a side channel")
            q = self.params['qrange_client_quantile']
            ema_at_quantile = torch.quantile(torch.tensor(self.ema_buffer), q).item()
            self.current_qrange = ema_at_quantile


class DynamicRangeQuantizationFederatedExperiment(
        QuantizationWithEqualBinsMixin,
        DynamicRangeMixin,
        BaseDigitalFederatedExperiment):
    """Digital federated experiment that quantizes each component of the model
    in a similar manner to `SimpleQuantizationFederatedExperiment`, but that
    dynamically adjusts the quantization range.
    """

    # For historical reasons (i.e., because this was originally part of
    # `DynamicRangeQuantizationFederatedExperiment`), the dynamic range is
    # called `qrange`.

    default_params = BaseDigitalFederatedExperiment.default_params.copy()
    default_params.update(QuantizationWithEqualBinsMixin.default_params_to_add)
    default_params.update(DynamicRangeMixin.default_params_to_add)

    def client_transmit(self, client: int, model: torch.nn.Module) -> torch.Tensor:
        values = self.get_values_to_send(model)
        self.update_qrange_buffer(client, values)

        lengths = self.bits_per_tx_parameter(client)
        assert values.shape == lengths.shape, f"shape mismatch: {values.shape} vs {lengths.shape}"
        qrange = self.current_qrange
        indices = self.quantize(values, lengths, qrange)
        return indices

    def server_receive(self, transmissions):
        qrange = self.current_qrange
        self.records['quantization_range'] = qrange  # take note of quantization range

        all_unquantized = []
        for i, indices in enumerate(transmissions):
            lengths = self.bits_per_tx_parameter(i)
            unquantized = self.unquantize(indices, lengths, qrange)
            all_unquantized.append(unquantized)

        client_average = self.compute_client_average(all_unquantized)
        self.update_global_model(client_average)
        self.update_qrange()


class DynamicRangeFederatedExperiment(DynamicRangeMixin, BaseFederatedExperiment):
    """Like `DynamicRangeQuantizationFederatedExperiment`, but doesn't do
    quantization. This is mostly for debugging/sanity checks.
    """

    default_params = BaseFederatedExperiment.default_params.copy()
    default_params.update(DynamicRangeMixin.default_params_to_add)

    def transmit_and_aggregate(self):
        """Receives the numbers errorlessly at the server and updates the model
        at the server.
        """
        transmissions = [self.client_transmit(i, model) for i, model in enumerate(self.client_models)]
        self.server_receive(transmissions)

    def client_transmit(self, client: int, model: torch.nn.Module) -> torch.Tensor:
        values = self.get_values_to_send(model)
        self.update_qrange_buffer(client, values)
        qrange = self.current_qrange
        clipped = values.clip(-qrange, qrange)
        return clipped

    def server_receive(self, transmissions):
        qrange = self.current_qrange
        self.records['dynamic_range'] = qrange  # take note of quantization range
        client_average = torch.stack(transmissions, 0).mean(0)
        self.update_global_model(client_average)
        self.update_qrange()
