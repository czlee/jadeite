"""Classes for analog federated experiments.

Experiment classes are supposed to be, as far as possible, agnostic to models,
loss functions and optimizers. They take care of training, testing and logging.
"""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import logging
from math import sqrt
from typing import Sequence

import torch

from .federated import BaseFederatedExperiment

logger = logging.getLogger(__name__)


class BaseOverTheAirExperiment(BaseFederatedExperiment):
    """Base class for over-the-air experiment.

    This class models our main proposed analog scheme. It trains clients, and
    then has clients transmit symbols over a simulated Gaussian MAC. The server
    updates the global model based on the noisy superposition of symbols that
    it receives.
    """

    default_params = BaseFederatedExperiment.default_params.copy()
    default_params.update({
        'noise': 1.0,
        'power': 1.0,
    })

    @classmethod
    def add_arguments(cls, parser):
        analog_args = parser.add_argument_group(title="Analog federated parameters")
        analog_args.add_argument("-N", "--noise", type=float,
            help="Noise level (variance), σₙ²")
        analog_args.add_argument("-P", "--power", type=float,
            help="Power level, P")

        super().add_arguments(parser)

    def channel(self, client_symbols: Sequence[torch.Tensor]) -> torch.Tensor:
        """Returns the channel output when the channel inputs are as provided in
        `client_symbols`, which should be a list of tensors.
        """
        σₙ = sqrt(self.params['noise'])  # stdev
        all_symbols = torch.vstack(client_symbols)
        sum_symbols = torch.sum(all_symbols, dim=0, keepdim=True)
        noise_sample = torch.normal(0.0, σₙ, size=sum_symbols.size()).to(self.device)
        output = sum_symbols + noise_sample
        assert output.dim() == 2 and output.size()[0] == 1
        return output

    def client_transmit(self, client: int, model: torch.nn.Module) -> torch.Tensor:
        """Returns the symbols that should be transmitted from the client with
        index `client`, as a row tensor. The `model` of the given client is also
        provided. Subclasses must implement this method.
        """
        raise NotImplementedError

    def server_receive(self, symbols):
        """Updates the global `model` given the `symbols` received from the
        channel. Subclasses must implement this method.
        """
        raise NotImplementedError

    def transmit_and_aggregate(self):
        """Transmits model data over the channel, receives a noisy version at
        the server and updates the model at the server."""
        tx_symbols = [self.client_transmit(i, model) for i, model in enumerate(self.client_models)]
        rx_symbols = self.channel(tx_symbols)
        self.server_receive(rx_symbols)


class OverTheAirExperiment(BaseOverTheAirExperiment):
    """Simple over-the-air experiment that just has the user specify the
    parameter radius ahead of time.

    In our initial experiments this seemed like a terrible idea in practice,
    because the values that need to be transmitted changes a lot depending on
    how far through training the model is. This, in turn, would see significant
    changes in actual power usage as training progressed, meaning in turn that
    the actual power used fell well short of the power constraint a lot of the
    time. A system that instead makes a simple effort to dynamically scale power
    is in DynamicPowerOverTheAirExperiment.
    """

    default_params = BaseOverTheAirExperiment.default_params.copy()
    default_params.update({
        'parameter_radius': 1.0,
    })

    @classmethod
    def add_arguments(cls, parser):
        ota_args = parser.add_argument_group(title="Basic over-the-air parameters")
        ota_args.add_argument("-B", "--parameter-radius", type=float,
            help="Parameter radius, B")

        super().add_arguments(parser)

    def client_transmit(self, client: int, model: torch.nn.Module) -> torch.Tensor:
        P = self.params['power']             # noqa: N806
        B = self.params['parameter_radius']  # noqa: N806

        values = self.get_values_to_send(model)
        symbols = values * sqrt(P) / B

        # record some statistics
        rms_value = sqrt(values.square().mean().item())
        self.records[f'msg_rms_client{client}'] = rms_value
        tx_power = symbols.square().mean().item()
        self.records[f"tx_power_client{client}"] = tx_power

        assert symbols.dim() == 2 and symbols.size()[0] == 1
        return symbols

    def server_receive(self, symbols):
        P = self.params['power']             # noqa: N806
        B = self.params['parameter_radius']  # noqa: N806

        scaled_symbols = symbols / self.nclients * B / sqrt(P)
        self.update_global_model(scaled_symbols)


class ExponentialMovingAverageMixin:
    """Mixin adding internal tracking of an exponential moving average of the
    root-mean-square of the transmitted vectors.

    The exponential moving average (EMA) coefficient is set by the
    `ema_coefficient` parameter, which should be between 0 and 1.

    Subclasses should call `self.update_ema_buffer(client, value)` to update the
    EMA. Typically, they'd do so just after calling `self.get_values_to_send()`
    (if the subclass inherits from `BaseFederatedExperiment`).

    This mixin just tracks the EMA. It's up to subclasses to figure out what to
    do with it. Subclasses can access the current EMAs in `self.ema_buffer`,
    which is a list of floats with one entry per client, corresponding to the
    client indices specified in the `self.update_ema_buffer(client, values)`
    call.

    Subclasses must set `self.nclients` (or inherit from a parent class that
    does so, like `BaseFederatedExperiment`) in its `__init__()`.

    This is used by `DynamicPowerOverTheAirExperiment` and
    `experiments.digital.DynamicRangeQuantizationFederatedExperiment`.

    (It would probably make sense for this to permit tracking several EMAs, but
    we don't have a need for this yet.)
    """

    default_params_to_add = {
        'ema_coefficient': 1 / 3,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check parameter values for sensibility
        if not (0.0 <= self.params['ema_coefficient'] <= 1.0):
            logger.warning("EMA coefficient should be between 0 and 1, found: "
                           f"{self.params['ema_coefficient']}")

        self.ema_buffer = [None] * self.nclients

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("-a", "--ema-coefficient", type=float, metavar="ALPHA",
            help="Exponential moving average coefficient, should be between 0 and 1")
        super().add_arguments(parser)

    def update_ema_buffer(self, client: int, value: float):
        if self.ema_buffer[client] is None:  # first value
            new_ema = value
        else:
            α = self.params['ema_coefficient']
            new_ema = α * value + (1 - α) * self.ema_buffer[client]

        self.ema_buffer[client] = new_ema
        self.records[f'ema_client{client}'] = new_ema


class DynamicPowerOverTheAirExperiment(ExponentialMovingAverageMixin, BaseOverTheAirExperiment):
    """Like OverTheAirExperiment, but this dynamically scales the parameter
    radius ("B") to try to maintain tx power at around the given power
    constraint, according to the following (very simple, presumptuous) protocol:

    Each client tracks an exponential moving average of the rms of the value
    vectors that it has transmitted. Every few (say, 5) periods, clients "send"
    their current moving average values to the server, which itself takes some
    percentile rank among the clients (say, the 90th percentile), divides it
    by some factor (say, 0.9), and sends that value back to the clients to be
    used by all clients as the parameter radius.

    The logic behind the percentile rank idea is that the power constraint is
    supposed to be satisfied by all clients individually, not just by the
    average among the clients. (Each client has its own battery!) We would use
    the 90th percentile (or something like that) just to avoid the effect of
    outliers getting in the way.

    This dynamic power control is hence governed by four parameters:
    - The exponential moving average coefficient is set by the `ema_coefficient`
      parameter.
    - The update frequency is set by the `power_update_period` parameter.
    - The percentile rank is set by the `power_quantile` parameter (and is
      actually between 0 and 1).
    - The divisor is set by the `power_factor` parameter.

    Since the parameter radius is dynamically controlled, this class does not
    have a `parameter_radius` parameter.
    """

    default_params = BaseOverTheAirExperiment.default_params.copy()
    default_params.update(ExponentialMovingAverageMixin.default_params_to_add)
    default_params.update({
        'power_update_period': 1,
        'power_quantile': 1.0,
        'power_factor': 0.9,
        'parameter_radius_initial': 1.0,
    })

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check parameter values for sensibility
        if not (0.0 <= self.params['power_quantile'] <= 1.0):
            logger.error("Dynamic power quantile must be between 0 and 1, found: "
                         f"{self.params['power_quantile']}")
        if not (0.0 <= self.params['power_factor'] <= 1.0):
            logger.warning("Dynamic power factor should normally be between 0 and 1, found: "
                           f"{self.params['power_factor']}")

        self.current_parameter_radius = self.params['parameter_radius_initial']

    @classmethod
    def add_arguments(cls, parser):
        power_args = parser.add_argument_group(title="Dynamic power control parameters")
        power_args.add_argument("-pup", "--power-update-period", type=int, metavar="PERIOD",
            help="Number of rounds between power control updates")
        power_args.add_argument("-pq", "--power-quantile", type=float, metavar="QUANTILE",
            help="Quantile among clients to take for power control, between 0 and 1 "
                 "(1 means take the maximum among clients)")
        power_args.add_argument("-pf", "--power-factor", type=float, metavar="FACTOR",
            help="Divide the inferred parameter radius by this value before use, should "
                 "generally be between 0 and 1 (normally closer to 1). This is called the "
                 "power factor because it has the effect of scaling the power by this factor, "
                 "so e.g. a factor of 0.8 would effectively scale down power by 20%%")
        power_args.add_argument("-Bin", "--parameter-radius-initial", type=float, metavar="B",
            help="Initial parameter radius, B (before dynamic power adjustment)")

        super().add_arguments(parser)

    def client_transmit(self, client: int, model: torch.nn.Module) -> torch.Tensor:
        values = self.get_values_to_send(model)

        rms_value = sqrt(values.square().mean().item())
        self.records[f'rms_client{client}'] = rms_value
        self.update_ema_buffer(client, rms_value)

        P = self.params['power']             # noqa: N806
        B = self.current_parameter_radius    # noqa: N806
        symbols = values * sqrt(P) / B

        # record some statistics
        tx_power = symbols.square().mean().item()
        self.records[f"tx_power_client{client}"] = tx_power

        assert symbols.dim() == 2 and symbols.size()[0] == 1
        return symbols

    def server_receive(self, symbols):
        P = self.params['power']             # noqa: N806
        B = self.current_parameter_radius    # noqa: N806

        self.records['parameter_radius'] = B  # take note of parameter radius

        scaled_symbols = symbols / self.nclients * B / sqrt(P)
        self.update_global_model(scaled_symbols)

        if self.current_round % self.params['power_update_period'] == 0:
            # update the current parameter radius by looking at all the clients
            # (i.e. the clients "sent this information through a side channel")
            q = self.params['power_quantile']
            γ = self.params['power_factor']
            ema_at_quantile = torch.quantile(torch.tensor(self.ema_buffer), q).item()
            self.current_parameter_radius = ema_at_quantile / sqrt(γ)
