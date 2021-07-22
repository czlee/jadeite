"""Classes for analog federated experiments.

Experiment classes are supposed to be, as far as possible, agnostic to models,
loss functions and optimizers. They take care of training, testing and logging.
"""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

from math import sqrt
from typing import Sequence

import torch

from .federated import BaseFederatedExperiment


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
        parser.add_argument("-N", "--noise", type=float,
            help="Noise level (variance), σₙ²")
        parser.add_argument("-P", "--power", type=float,
            help="Power level, P")

        super().add_arguments(parser)

    def get_parameter_radius(self):
        raise NotImplementedError

    def client_transmit(self, model) -> torch.Tensor:
        """Returns the symbols that should be transmitted from the client that
        is working with the given (client) `model`, as a row tensor.
        """
        P = self.params['power']             # noqa: N806
        B = self.get_parameter_radius()      # noqa: N806
        values = self.get_values_to_send(model)
        symbols = values * sqrt(P) / B
        assert symbols.dim() == 2 and symbols.size()[0] == 1
        return symbols

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

    def server_receive(self, symbols):
        """Updates the global `model` given the `symbols` received from the
        channel.
        """
        P = self.params['power']             # noqa: N806
        B = self.get_parameter_radius()      # noqa: N806

        scaled_symbols = symbols / self.nclients * B / sqrt(P)
        self.update_global_model(scaled_symbols)

    def record_tx_powers(self, tx_symbols) -> dict:
        records = {}
        for i, symbols in enumerate(tx_symbols):
            tx_power = (symbols.square().sum().cpu() / symbols.numel()).numpy()
            records[f"tx_power_client{i}"] = tx_power
        return records

    def transmit_and_aggregate(self, records: dict):
        """Transmits model data over the channel, receives a noisy version at
        the server and updates the model at the server."""
        tx_symbols = [self.client_transmit(model) for model in self.client_models]
        records.update(self.record_tx_powers(tx_symbols))
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
        parser.add_argument("-B", "--parameter-radius", type=float,
            help="Parameter radius, B")

        super().add_arguments(parser)

    def get_parameter_radius(self):
        return self.params['parameter_radius']


class DynamicPowerOverTheAirExperiment(BaseOverTheAirExperiment):
    """Like OverTheAirExperiment, but this dynamically scales the parameter
    radius ("B") to try to maintain tx power at around the given power
    constraint, according to the following (very simple, presumptuous) protocol:

    Each client tracks a moving average of the norm of the value vectors that it
    has transmitted. Every few (say, 5) periods, clients "send" their current
    moving average values to the server, which itself takes some percentile rank
    among the clients (say, the 90th percentile), multiplies it by some factor
    (say, 0.9), and sends that value back to the clients to be used by all
    clients as the parameter radius.

    The logic behind the percentile rank idea is that the power constraint is
    supposed to be satisfied by all clients individually, not just by the
    average among the clients. (Each client has its own battery!) We would use
    the 90th percentile (or something like that) just to avoid the effect of
    outliers getting in the way.

    This dynamic power control is hence governed by four parameters:
    - The moving average window is set by the `power_average_period` parameter.
    - The update frequency is set by the `power_update_period` parameter.
    - The percentile rank is set by the `power_quantile` parameter (and is
      actually between 0 and 1).
    - The multiplier is set by the `power_multiplier` parameter.

    Since the parameter radius is dynamically controlled, this class does not
    have a `parameter_radius` parameter.
    """

    default_params = BaseFederatedExperiment.default_params.copy()
    default_params.update({
        'power_average_period': 5,
        'power_update_period': 5,
        'power_quantile': 0.9,
        'power_multiplier': 0.9,
    })

    @classmethod
    def add_arguments(cls, parser):
        power_args = parser.add_argument_group(title="Dynamic power control parameters")
        power_args.add_argument("-pap", "--power-average-period", type=int, metavar="PERIOD",
            help="Moving average period for dynamic power control, in number of rounds")
        power_args.add_argument("-pup", "--power-update-period", type=int, metavar="PERIOD",
            help="Number of rounds between power control updates")
        power_args.add_argument("-pq", "--power-quantile", type=float, metavar="QUANTILE",
            help="Quantile among clients to take for power control")
        power_args.add_argument("-pm", "--power-multiplier", type=float, metavar="MULTIPLIER",
            help="Multiply the inferred parameter radius by this value before use")

        super().add_arguments(parser)
