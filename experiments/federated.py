"""Classes for federated experiments.

Experiment classes are supposed to be, as far as possible, agnostic to models,
loss functions and optimizers. They take care of training, testing and logging.

Note on parameter/`state_dict` distinction: Currently, this implementation
treats all members of the `state_dict` as parameters that need to be
communicated over the network, whether or not they are model parameters (or e.g.
buffers). It seems sensible to me to sync all of the `state_dict`, but I'm not
100% sure if this is actually what needs to happen, so if I later discover that
this is mistaken, this implementation may change to send only elements in
`model.parameters()`.
"""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import argparse
import logging
import pathlib
from typing import Callable, Dict, Sequence

import torch

import utils

from .experiment import BaseExperiment

logger = logging.getLogger(__name__)


class BaseFederatedExperiment(BaseExperiment):
    """Base class for federated experiments.

    This takes care of splitting the datasets among clients and training
    individual clients, which should be common functionality to all federated
    experiments.
    """

    default_params = BaseExperiment.default_params.copy()
    default_params.update({
        'epochs': 1,
        'rounds': 20,
        'clients': 10,
        'send': 'deltas',
    })

    def __init__(
            self,
            client_datasets: Sequence[torch.utils.data.Dataset],
            test_dataset: torch.utils.data.Dataset,
            client_models: Sequence[torch.nn.Module],
            global_model: torch.nn.Module,
            loss_fn: Callable,
            metric_fns: Dict[str, Callable],
            client_optimizers: Sequence[torch.optim.Optimizer],
            results_dir: pathlib.Path,
            device='cpu',
            **params):
        """This constructor requires all client datasets, models and optimizers
        to be pre-constructed ready to be passed into this constructor. The
        constructor will sync the client models with the global model before
        starting training.
        """
        super().__init__(loss_fn, metric_fns, results_dir, device, **params)

        if not len(client_datasets) == len(client_models) == len(client_optimizers):
            raise ValueError(f"There are {len(client_datasets)} client datasets, "
                             f"{len(client_models)} client models and "
                             f"{len(client_optimizers)} client optimizers.")

        self.nclients = len(client_datasets)
        self.client_datasets = client_datasets
        self.test_dataset = test_dataset
        self.client_models = [model.to(device) for model in client_models]
        self.global_model = global_model.to(device)
        self.client_optimizers = client_optimizers

        self.client_dataloaders = [
            torch.utils.data.DataLoader(dataset, batch_size=self.params['batch_size'])
            for dataset in self.client_datasets
        ]

        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.params['batch_size'],
        )

        # sync client models before starting
        for model in self.client_models:
            model.load_state_dict(global_model.state_dict())

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):

        federated_args = parser.add_argument_group("Federated learning parameters")
        federated_args.add_argument("-r", "--rounds", type=int,
            help="Number of rounds")
        federated_args.add_argument("-n", "--clients", type=int,
            help="Number of clients, n")
        federated_args.add_argument("-l", "--lr-client", type=float, default=1e-2,
            help="Learning rate at client")
        federated_args.add_argument("-dpc", "--data-per-client", type=int, default=None,
            help="Override the number of data points each client has (default: "
                 "divide all data equally among clients)")
        federated_args.add_argument("--send", choices=["params", "deltas"],
            help="What clients should send. 'params' sends the model parameters; "
                 "'deltas' sends additive updates to model parameters.")

        super().add_arguments(parser)

    @classmethod
    def from_arguments(cls,
            train_dataset: Sequence[torch.utils.data.Dataset],
            test_dataset: torch.utils.data.Dataset,
            model_fn: Callable[[], torch.nn.Module],
            loss_fn: Callable,
            metric_fns: Dict[str, Callable],
            results_dir: pathlib.Path,
            args: argparse.Namespace):
        """Instantiates a FederatedAveragingExperiment object from arguments
        provided by an `ArgumentParser.parse_args()` call.
        """
        device = cls._interpret_cpu_arg(args.cpu)
        nclients = args.clients

        data_per_client = args.data_per_client
        if data_per_client is None:
            client_lengths = utils.divide_integer_evenly(len(train_dataset), nclients)
        else:
            if data_per_client * nclients > len(train_dataset):
                raise ValueError(f"There isn't enough data ({len(train_dataset)}) to get "
                                 f"{data_per_client} examples for each of {nclients} clients.")
            client_lengths = [data_per_client] * nclients
            train_dataset = torch.utils.data.Subset(train_dataset, range(data_per_client * nclients))

        client_datasets = torch.utils.data.random_split(train_dataset, client_lengths)
        global_model = model_fn()
        client_models = [model_fn() for i in range(nclients)]

        client_optimizers = [
            torch.optim.SGD(model.parameters(), lr=args.lr_client)
            for model in client_models
        ]

        params = cls.extract_params_from_args(args)

        return cls(client_datasets, test_dataset, client_models, global_model, loss_fn, metric_fns,
                   client_optimizers, results_dir, device, **params)

    def train_clients(self):
        """Trains all clients through one round of the number of epochs specified
        in `self.params['epochs']`.
        """
        records = {}
        clients = zip(self.client_dataloaders, self.client_models, self.client_optimizers)
        nepochs = self.params['epochs']

        for i, (dataloader, model, optimizer) in enumerate(clients):
            for j in range(nepochs):
                train_loss = self._train(dataloader, model, optimizer)
                logger.info(f"Client {i}/{self.nclients}, epoch {j}/{nepochs}: loss {train_loss}")
            records[f"train_loss_client{i}"] = train_loss

        return records

    @staticmethod
    def flatten_state_dict(state_dict: dict) -> torch.Tensor:
        """Flattens a given model's state dict into a single tensor. Normally,
        the state dict passed to this method will be that of a client model.

        Subclasses normally shouldn't use this method. To retrieve the values
        the client should send, use `get_values_to_send()`.
        """
        states = [state.flatten() for state in state_dict.values()]
        flattened = torch.hstack(states).reshape(1, -1)
        return flattened

    def unflatten_state_dict(self, tensor: torch.Tensor) -> dict:
        """Unflattens a (presumably 1-D) tensor into a state dict compatible
        with the global model.

        Subclasses normally shouldn't use this method. To update the model with
        received values, use `update_global_model()`.
        """
        flattened = tensor.flatten()
        new_state_dict = {}
        cursor = 0
        for key, value in self.global_model.state_dict().items():
            numel = value.numel()
            part = flattened[cursor:cursor + numel]
            new_state_dict[key] = part.reshape(value.size())
            cursor += numel
        assert cursor == flattened.numel()
        return new_state_dict

    def get_values_to_send(self, model) -> torch.Tensor:
        """Returns the values that should be sent from the client.

        Subclasses can use this method to retrieve the vector that needs to be
        communicated from clients to the server."""
        local_flattened = self.flatten_state_dict(model.state_dict())

        if self.params['send'] == 'deltas':
            global_flattened = self.flatten_state_dict(self.global_model.state_dict())
            return local_flattened - global_flattened

        elif self.params['send'] == 'params':
            return local_flattened

        else:
            raise ValueError("Unknown 'send' spec: " + str(self.params['send']))

    def update_global_model(self, values):
        """Update the global model with the values provided, which should be the
        values inferred by the server from the received signals. For example, in
        federated averaging, `values` would be the mean of what each client
        returns from `get_values_to_send()`.

        Subclasses can use this method to handle received values.
        """
        if self.params['send'] == 'deltas':
            global_flattened = self.flatten_state_dict(self.global_model.state_dict())
            updated_values = global_flattened + values
            new_state_dict = self.unflatten_state_dict(updated_values)

        elif self.params['send'] == 'params':
            new_state_dict = self.unflatten_state_dict(values)

        self.global_model.load_state_dict(new_state_dict)

    def test(self):
        return self._test(self.test_dataloader, self.global_model)

    def transmit_and_aggregate(self):
        """Transmits the client models from `self.client_models` and aggregates
        them at the server. At the time this is called, the clients are assumed
        to have been trained for this round (by `self.train_clients()`).

        When this is called, `self.records` is a dict that will be logged to a
        CSV file (with keys as column headers). Subclasses may optionally add
        entries to this dict. If they do so, they should modify the dict
        in-place. Also, `self.current_round` will be the current round number.

        Subclasses must implement this method."""
        raise NotImplementedError

    def run(self):
        """Runs the experiment once."""
        nrounds = self.params['rounds']
        csv_logger = self.get_csv_logger('training.csv', index_field='round')

        for r in range(nrounds):
            self.current_round = r

            self.records = self.train_clients()  # this overwrites self.records
            self.transmit_and_aggregate()

            test_results = self.test()
            self.records.update(test_results)

            logger.info(f"Round {r}: " + ", ".join(f"{k} {v:.7f}" for k, v in test_results.items()))
            csv_logger.log(r, self.records)
            self.log_model_json(r, self.global_model)

        self.current_round = None
        csv_logger.close()

        test_results = self.test()
        self.log_evaluation(test_results)


class OldFederatedAveragingExperiment(BaseFederatedExperiment):
    """Old version of a class for a simple federated averaging experiment.

    This is a simpler version of `FederatedAveragingExperiment`. Rather than
    flatten and unflatten the state dict, it goes through the state dict and
    averages each tensor in the state dict separately. It's deprecated in favor
    of `FederatedAveragingExperiment`."""

    def __init__(self, *args, **params):
        super().__init__(*args, **params)
        if self.params['send'] != 'params':
            logger.error("OldFederatedAveragingExperiment only supports sending model parameters.")
            logger.error("Use the new FederatedAveragingExperiment to send deltas.")
            raise ValueError("OldFederatedAveragingExperiment called with sending deltas")

    def transmit_and_aggregate(self):
        """Aggregates client models by taking the mean."""
        global_dict = self.global_model.state_dict()
        for k in global_dict.keys():
            client_states = [model.state_dict()[k].float() for model in self.client_models]
            global_dict[k] = torch.stack(client_states, 0).mean(0)

        self.global_model.load_state_dict(global_dict)
        for model in self.client_models:
            model.load_state_dict(self.global_model.state_dict())


class FederatedAveragingExperiment(BaseFederatedExperiment):
    """Class for a simple federated averaging experiment.

    This class doesn't attempt to model the channel at all. It just trains
    clients individually, and assumes the clients can send whatever they want
    to the server errorlessly.

    This class should do the same thing as `OldFederatedAveragingExperiment`,
    just in a slightly more roundabout way. It uses the `get_values_to_send()`
    and `update_global_model()` methods of BaseFederatedExperiment to simplify
    its own implementation. The advantage of doing this is that the class can
    take advantage of options specifying what values clients should send (e.g.,
    whether to send the model parameters themselves, or updates as deltas). The
    disadvantage is that flattening and unflattening the state dict is, overall,
    a little bit more complicated.
    """

    def transmit_and_aggregate(self):
        """Aggregates client models."""
        client_values = [self.get_values_to_send(model) for model in self.client_models]
        client_average = torch.stack(client_values, 0).mean(0)
        self.update_global_model(client_average)
