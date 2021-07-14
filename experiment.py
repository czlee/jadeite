"""Class for a single experiment.

Experiment classes are supposed to be, as far as possible, agnostic to models,
loss functions and optimizers. They take care of training, testing and logging.
"""

import argparse
import json
import pathlib
import time
from typing import Callable, Dict, Sequence

import torch

import data.utils
import results


class BaseExperiment:
    """Base class for experiments."""

    default_params = {
        'epochs': 20,
        'batch_size': 64,
    }

    def __init__(self,
            loss_fn: Callable,
            metric_fns: Dict[str, Callable],
            results_dir: pathlib.Path,
            device='cpu',
            **params):
        """This constructor just sets up attributes that are common to all
        experiments. It's not many, and in particular omits things like datasets
        and models, how these work differ between non-federated and federated
        experiments.
        """
        self.loss_fn = loss_fn
        self.metric_fns = metric_fns
        self.device = device
        self.results_dir = results_dir

        # check for metric names that would conflict 'train_loss'
        for m in ['train_loss', 'test_loss', 'epoch', 'timestamp', 'finished']:
            assert m not in metric_fns, f"{m} can't be used as a metric name"

        # set up parameters
        self.params = self.default_params.copy()
        self.params.update(params)
        unexpected_params = set(self.params.keys()) - set(self.default_params.keys())
        if unexpected_params:
            print(f"Warning: Unexpected parameters {unexpected_params}")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Adds relevant command-line arguments to the given `parser`.
        Subclasses should call `super()` if they override this, and `super()`
        should be called at the end of the implementation, because the last
        thing it does is set the argument defaults to match
        `cls.default_params`.
        """
        parser.add_argument("-e", "--epochs", type=int,
            help="Number of epochs")
        parser.add_argument("-b", "--batch-size", type=int,
            help="Batch size")
        parser.add_argument("--cpu", action="store_true", default=False,
            help="Force use of CPU (i.e. don't use CUDA)")
        parser.set_defaults(**cls.default_params)

    def log_model_json(self, seq, model):
        """Writes the model parameters of the given model to a file in the
        results directory. `seq` should be a sequence number, typically the
        epoch or round number, but it can be anything that is compatible with
        a file name.
        """
        model_file = self.results_dir / f"model_at_{seq}.json"
        with open(model_file, 'w') as f:
            states = {k: v.tolist() for k, v in model.state_dict().items()}
            json.dump(states, f, indent=2)

    def log_evaluation(self, evaluation_dict):
        results.log_evaluation(evaluation_dict, self.results_dir)

    def get_csv_logger(self, filename, **kwargs):
        logfile = self.results_dir / 'training.csv'
        return results.CsvLogger(logfile, **kwargs)

    def _train(self, dataloader, model, optimizer):
        """Trains through one epoch. The dataloader, model and optimizer need to
        be provided, so subclasses will probably want to implement a `train()`
        method that calls this. The loss function and metrics are taken from the
        instance.
        """
        model.train()
        nbatches = len(dataloader)
        last_update_time = time.time()

        for batch, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)

            pred = model(x)
            loss = self.loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            now = time.time()
            if now - last_update_time > 1:
                print(f"[{batch}/{nbatches} {(batch/nbatches):.0%}] loss: {loss.item()}...", end='\r')

        return loss.item()

    def _test(self, dataloader, model):
        """Evaluates the current model on the test dataset, using both loss and
        all metrics from the instance. The dataloader and model need to be
        provided, so subclasses will probably want to implement a `test()`
        method that calls this.
        """
        model.eval()

        results = dict.fromkeys(self.metric_fns.keys(), 0)
        results['test_loss'] = 0

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                pred = model(x)
                results['test_loss'] += self.loss_fn(pred, y).item()

                for metric_name, metric_fn in self.metric_fns.items():
                    results[metric_name] += metric_fn(pred, y).item()

        for metric_name in results.keys():
            results[metric_name] /= len(dataloader)

        return results


class SimpleExperiment(BaseExperiment):
    """Class for a simple non-federated experiment."""

    default_params = BaseExperiment.default_params.copy()

    def __init__(
            self,
            train_dataset: torch.utils.data.Dataset,
            test_dataset: torch.utils.data.Dataset,
            model: torch.nn.Module,
            loss_fn: Callable,
            metric_fns: Dict[str, Callable],
            optimizer: torch.optim.Optimizer,
            results_dir: pathlib.Path,
            device='cpu',
            **params):

        super().__init__(loss_fn, metric_fns, results_dir, device, **params)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model.to(device)
        self.optimizer = optimizer

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.params['batch_size'],
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.params['batch_size'],
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Adds relevant command-line arguments to the given `parser`.
        Subclasses should call super() if they override this.
        """
        parser.add_argument("-l", "--lr", "--learning-rate", type=float, default=1e-2,
            help="Learning rate")
        super().add_arguments(parser)

    @classmethod
    def from_arguments(cls,
            train_dataset: torch.utils.data.Dataset,
            test_dataset: torch.utils.data.Dataset,
            model: torch.nn.Module,
            loss_fn: Callable,
            metric_fns: Dict[str, Callable],
            results_dir: pathlib.Path,
            args: argparse.Namespace):
        """Instantiates a SimpleExperiment object from arguments provided by an
        `ArgumentParser.parse_args()` call.

        The datasets, model, loss and metrics are still directly specified by
        the caller --- they're too complicated to try to "generalize".
        """
        device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        params = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
        }
        return cls(train_dataset, test_dataset, model, loss_fn, metric_fns,
                   optimizer, results_dir, device, **params)

    def run(self):
        """Runs the experiment once."""
        nepochs = self.params['epochs']
        logger = self.get_csv_logger('training.csv')

        for i in range(nepochs):
            print(f"Epoch {i} of {nepochs}...", end='\r')
            train_loss = self.train()
            test_results = self.test()
            test_results['train_loss'] = train_loss
            print(f"Epoch {i}: " + ", ".join(f"{k} {v:.7f}" for k, v in test_results.items()))
            logger.log(i, test_results)
            self.log_model_json(i, self.model)

        logger.close()
        test_results = self.test()
        self.log_evaluation(test_results)

    def train(self):
        return self._train(self.train_dataloader, self.model, self.optimizer)

    def test(self):
        return self._test(self.test_dataloader, self.model)


class FederatedAveragingExperiment(BaseExperiment):

    default_params = BaseExperiment.default_params.copy()
    default_params.update({
        'epochs': 1,
        'rounds': 20,
        'clients': 10,
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
        to be pre-constructed ready to be passed into this constructor.

        The constructor will sync the client models with the global model before
        starting training.
        """
        super().__init__(loss_fn, metric_fns, results_dir, device, **params)

        if not len(client_datasets) == len(client_models) == len(client_optimizers):
            raise ValueError(f"There are {len(client_datasets)} client datasets, "
                             f"{len(client_models)} client models and "
                             f"{len(client_optimizers)} client optimizers.")

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
    def add_arguments(cls, parser):
        """Adds relevant command-line arguments to the given `parser`, which
        should be an `argparse.ArgumentParser` object.
        """
        parser.add_argument("-r", "--rounds", type=int, default=20,
            help="Number of rounds")
        parser.add_argument("-c", "--clients", type=int, default=10,
            help="Number of clients")
        parser.add_argument("-l", "--lr-client", "--learning-rate-client",
            type=float, default=1e-2, help="Learning rate at client")

        super().add_arguments(parser)

    @classmethod
    def from_arguments_divide_evenly(cls,
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
        device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        nclients = args.clients

        client_lengths = data.utils.divide_integer_evenly(len(train_dataset), nclients)
        client_datasets = torch.utils.data.random_split(train_dataset, client_lengths)
        global_model = model_fn()
        client_models = [model_fn() for i in range(nclients)]

        client_optimizers = [
            torch.optim.SGD(model.parameters(), lr=args.lr_client)
            for model in client_models
        ]

        params = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'rounds': args.rounds,
        }

        return cls(client_datasets, test_dataset, client_models, global_model, loss_fn, metric_fns,
                   client_optimizers, results_dir, device, **params)

    def train_clients(self):
        """Trains all clients through one round of the number of epochs specified
        in `self.params['epochs']`.
        """
        records = {}
        clients = zip(self.client_dataloaders, self.client_models, self.client_optimizers)
        nclients = len(self.client_dataloaders)
        nepochs = self.params['epochs']

        for i, (dataloader, model, optimizer) in enumerate(clients):
            for j in range(nepochs):
                train_loss = self._train(dataloader, model, optimizer)
                print(f"Client {i}/{nclients}, epoch {j}/{nepochs}: loss {train_loss}")
            records[f"train_loss_client{i}"] = train_loss

        return records

    def server_aggregate(self):
        """Aggregates client models by taking the mean."""
        global_dict = self.global_model.state_dict()
        for k in global_dict.keys():
            client_states = [model.state_dict()[k].float() for model in self.client_models]
            global_dict[k] = torch.stack(client_states, 0).mean(0)

        self.global_model.load_state_dict(global_dict)
        for model in self.client_models:
            model.load_state_dict(self.global_model.state_dict())

    def test(self):
        return self._test(self.test_dataloader, self.global_model)

    def run(self):
        """Runs the experiment once."""
        nrounds = self.params['rounds']
        logger = self.get_csv_logger('training.csv', index_field='round')

        for r in range(nrounds):
            records = self.train_clients()
            self.server_aggregate()
            test_results = self.test()
            records.update(test_results)

            print(f"Round {r}: " + ", ".join(f"{k} {v:.7f}" for k, v in test_results.items()))
            logger.log(r, records)
            self.log_model_json(r, self.global_model)

        logger.close()
        test_results = self.test()
        self.log_evaluation(test_results)


class OverTheAirExperiment(SimpleExperiment):

    default_params = SimpleExperiment.default_params.copy()
    default_params.update({
        'epochs': 1,
        'noise': 1.0,
        'power': 1.0,
        'parameter_radius': 1.0,
    })
