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


class OverTheAirExperiment(BaseFederatedExperiment):
    """Class for over-the-air experiment.

    This class models our main proposed analog scheme. It trains clients, and
    then has clients transmit symbols over a simulated Gaussian MAC. The server
    updates the global model based on the noisy superposition of symbols that
    it receives.
    """

    default_params = BaseFederatedExperiment.default_params.copy()
    default_params.update({
        'noise': 1.0,
        'power': 1.0,
        'parameter_radius': 1.0,
    })

    @classmethod
    def add_arguments(cls, parser):
        """Adds relevant command-line arguments to the given `parser`, which
        should be an `argparse.ArgumentParser` object.
        """
        parser.add_argument("-N", "--noise", type=float,
            help="Noise level (variance), σₙ²")
        parser.add_argument("-P", "--power", type=float,
            help="Power level, P")
        parser.add_argument("-B", "--parameter-radius", type=float,
            help="Parameter radius, B")

        super().add_arguments(parser)

    def client_transmit(self, model) -> torch.Tensor:
        """Returns the symbols that should be transmitted from the client that
        is working with the given (client) `model`, as a row tensor.
        """
        P = self.params['power']             # noqa: N806
        B = self.params['parameter_radius']  # noqa: N806
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
        B = self.params['parameter_radius']  # noqa: N806

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
