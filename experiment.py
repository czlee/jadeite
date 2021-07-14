"""Class for a single experiment.

Experiment classes are supposed to be, as far as possible, agnostic to models,
loss functions and optimizers. They take care of training, testing and logging.
"""

import json
import pathlib
import time
from typing import Callable, Dict

import torch

import results


class SimpleExperiment:
    """Class for a single experiment, by which we mean a single attempt to train
    a model."""

    default_params = {
        'epochs': 20,
        'batch_size': 64,
    }

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

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.metric_fns = metric_fns
        self.optimizer = optimizer
        self.device = device
        self.results_dir = results_dir

        # check for metric names that would conflict 'train_loss'
        for m in ['train_loss', 'test_loss', 'epoch', 'timestamp']:
            assert m not in metric_fns, f"{m} can't be used as a metric name"

        self.params = self.default_params.copy()
        self.params.update(params)
        unexpected_params = set(self.params.keys()) - set(self.default_params.keys())
        if unexpected_params:
            print(f"Warning: Unexpected parameters {unexpected_params}")

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.params['batch_size'],
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.params['batch_size'],
        )

    @classmethod
    def add_arguments(cls, parser):
        """Adds relevant command-line arguments to the given parser.
        Subclasses should call super() if they override this."""
        parser.add_argument("-e", "--epochs", type=int, default=cls.default_params['epochs'],
            help="Number of epochs")
        parser.add_argument("-b", "--batch-size", type=int, default=cls.default_params['batch_size'],
            help="Batch size")
        parser.add_argument("-l", "--lr", "--learning-rate", type=float, default=1e-2,
            help="Learning rate")
        parser.add_argument("--cpu", action="store_true", default=False,
            help="Force use of CPU (i.e. don't use CUDA)")

    @classmethod
    def from_arguments(cls, datasets, model, loss_fn, metric_fns, results_dir, args):
        """Instantiates an Experiment object from arguments provided by an
        `ArgumentParser.parse_args()` call.

        The datasets, model, loss and metrics are still directly specified by
        the caller --- they're too complicated to try to "generalize".
        """
        train_dataset, test_dataset = datasets
        device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        params = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
        }
        return cls(train_dataset, test_dataset, model, loss_fn, metric_fns,
                   optimizer, results_dir, device, **params)

    def run(self):
        nepochs = self.params['epochs']
        logfile = self.results_dir / 'training.csv'
        logger = results.CsvLogger(logfile)

        for i in range(nepochs):
            print(f"Epoch {i} of {nepochs}...", end='\r')
            train_loss = self.train()
            test_results = self.test()
            test_results['train_loss'] = train_loss
            print(f"Epoch {i}: " + ", ".join(f"{k} {v}" for k, v in test_results.items()))
            logger.log(i, test_results)
            self.log_model_json(i, self.model)

        test_results = self.test()
        results.log_evaluation(test_results, self.results_dir)

    def log_model_json(self, i, model):
        model_file = self.results_dir / f"model_at_{i}.json"
        with open(model_file, 'w') as f:
            states = {k: v.tolist() for k, v in model.state_dict().items()}
            json.dump(states, f, indent=2)

    def train(self):
        """Trains through one epoch."""
        self.model.train()

        nbatches = len(self.train_dataloader)

        last_update_time = time.time()

        for batch, (x, y) in enumerate(self.train_dataloader):
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.model(x)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            now = time.time()
            if now - last_update_time > 1:
                print(f"[{batch}/{nbatches} {(batch/nbatches):.0%}] loss: {loss.item()}...", end='\r')

        return loss.item()

    def test(self):
        """Evaluates the current model on the test dataset, using both loss and
        all metrics specified in `metric_fns`."""
        self.model.eval()

        results = dict.fromkeys(self.metric_fns.keys(), 0)
        results['test_loss'] = 0

        with torch.no_grad():
            for x, y in self.test_dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.model(x)
                results['test_loss'] += self.loss_fn(pred, y).item()

                for metric_name, metric_fn in self.metric_fns.items():
                    results[metric_name] += metric_fn(pred, y).item()

        for metric_name in results.keys():
            results[metric_name] /= len(self.test_dataloader)

        return results


class OverTheAirExperiment(SimpleExperiment):

    default_params = SimpleExperiment.default_params.copy()
    default_params.update({
        'rounds': 20,
        'epochs': 1,
        'clients': 10,
        'lr_client': 1e-2,
        'noise': 1.0,
        'power': 1.0,
        'parameter_radius': 1.0,
    })
