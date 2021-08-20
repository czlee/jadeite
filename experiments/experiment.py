"""Classes for a single experiment.

Experiment classes are supposed to be, as far as possible, agnostic to models,
loss functions and optimizers. They take care of training, testing and logging.
"""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import argparse
import json
import logging
import pathlib
import time
from typing import Callable, Dict

import torch

import utils

logger = logging.getLogger(__name__)


class BaseExperiment:
    """Base class for experiments."""

    default_params = {
        'epochs': 20,
        'batch_size': 64,
        'save_models': False,
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
            logger.warning(f"Unexpected parameters {unexpected_params}")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Adds relevant command-line arguments to the given `parser`.

        Subclasses should call `super().add_arguments()` if they override this,
        and this should be called at the end of the implementation, because the
        last thing it does is set the argument defaults to match
        `cls.default_params`. Accordingly, there is no need to specify the
        defaults in `parser.add_argument()` calls if they are set in
        `cls.default_params`.
        """
        parser.add_argument("-e", "--epochs", type=int,
            help="Number of epochs")
        parser.add_argument("-b", "--batch-size", type=int,
            help="Batch size")
        parser.add_argument("--cpu", action="store_true",
            help="Force use of CPU, i.e. don't use CUDA even if available")
        parser.add_argument("--save-models", action="store_true",
            help="Save JSON files with model parameters after every round "
                 "(warning: this uses up a lot of disk space)")

        # Set defaults, but only if it's in both cls.default_params and the parser's arguments
        arg_names = vars(parser.parse_args([])).keys()  # get all existing arguments
        common_names = set(arg_names) & set(cls.default_params.keys())
        defaults = {key: cls.default_params[key] for key in common_names}
        parser.set_defaults(**defaults)

    @staticmethod
    def _interpret_cpu_arg(use_cpu):
        """Returns 'cpu' or 'cuda' to indicate an argument to `device=` of
        pytorch functions. If `use_cpu` is False but CUDA isn't available, this
        logs a warning. Intended to be used by subclasses in `from_arguments()`
        constructors.
        """
        if use_cpu:
            return 'cpu'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            logger.warning("CUDA not available, using CPU instead")
            return 'cpu'

    @classmethod
    def extract_params_from_args(cls, args: argparse.Namespace, ignore_missing=[]):
        """Extracts parameters from arguments in an `argparse.Namespace` object.
        Subclasses can (and should) use this to process command-line arguments,
        and the dict returned by this function should be passed into the
        constructor as the **params dict.
        """
        params = {}
        args_dict = vars(args)
        for key in cls.default_params.keys():
            if key in args_dict:
                params[key] = args_dict[key]
            elif key not in ignore_missing:
                logger.warning(f"Parameter not found in arguments passed to {cls.__name__}: {key}")
        logger.debug("Extracted parameters: " + str(params))
        return params

    def log_model_json(self, seq, model, prefix=""):
        """Writes the model parameters of the given model to a file in the
        results directory. `seq` should be a sequence number, typically the
        epoch or round number, but it can be anything that is compatible with
        a file name.
        """
        if not self.params['save_models']:
            return

        model_file = self.results_dir / f"{prefix}model_at_{seq}.json"
        with open(model_file, 'w') as f:
            states = {k: v.tolist() for k, v in model.state_dict().items()}
            json.dump(states, f, indent=2)

    def log_evaluation(self, evaluation_dict):
        utils.log_evaluation(evaluation_dict, self.results_dir)

    def get_csv_logger(self, filename, **kwargs):
        logfile = self.results_dir / 'training.csv'
        return utils.CsvLogger(logfile, **kwargs)

    def show_progress(self, *args, **kwargs):
        if logger.getEffectiveLevel() <= logging.INFO:
            print(*args, **kwargs)

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
            if now - last_update_time > 0.4:
                self.show_progress(f"training [{batch}/{nbatches} {(batch/nbatches):.0%}] "
                                   f"loss: {loss.item():.7f}...", end='\r')
                last_update_time = now

        return loss.item()

    def _test(self, dataloader, model):
        """Evaluates the current model on the test dataset, using both loss and
        all metrics from the instance. The dataloader and model need to be
        provided, so subclasses will probably want to implement a `test()`
        method that calls this.
        """
        model.eval()
        nbatches = len(dataloader)
        last_update_time = time.time()

        results = dict.fromkeys(self.metric_fns.keys(), 0)
        results['test_loss'] = 0

        with torch.no_grad():
            for batch, (x, y) in enumerate(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)

                pred = model(x)
                results['test_loss'] += self.loss_fn(pred, y).item()

                for metric_name, metric_fn in self.metric_fns.items():
                    results[metric_name] += metric_fn(pred, y).item()

                now = time.time()
                if now - last_update_time > 0.4:
                    self.show_progress(f"testing [{batch}/{nbatches} {(batch/nbatches):.0%}] ", end='\r')
                    last_update_time = now

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
        optimizer_args = parser.add_argument_group("Optimizer parameters")
        optimizer_args.add_argument("-l", "-lr", "--learning-rate", type=float, default=1e-2,
            help="Learning rate")
        optimizer_args.add_argument("-mom", "--momentum", type=float, default=0.0,
            help="Momentum of SGD")
        optimizer_args.add_argument("-wd", "--weight-decay", type=float, default=0.0,
            help="Weight decay of SGD")
        super().add_arguments(parser)

    @classmethod
    def from_arguments(cls,
            train_dataset: torch.utils.data.Dataset,
            test_dataset: torch.utils.data.Dataset,
            model_fn: Callable[[], torch.nn.Module],
            loss_fn: Callable,
            metric_fns: Dict[str, Callable],
            results_dir: pathlib.Path,
            args: argparse.Namespace):
        """Instantiates a SimpleExperiment object from arguments provided by an
        `ArgumentParser.parse_args()` call.

        The datasets, model, loss and metrics are still directly specified by
        the caller --- they're too complicated to try to "generalize".
        """
        model = model_fn()
        device = cls._interpret_cpu_arg(args.cpu)
        logger.debug(f"Optimizer arguments: lr {args.learning_rate}, momentum {args.momentum}, "
                     f"weight decay {args.weight_decay}")
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        params = cls.extract_params_from_args(args)
        return cls(train_dataset, test_dataset, model, loss_fn, metric_fns,
                   optimizer, results_dir, device, **params)

    def run(self):
        """Runs the experiment once."""
        nepochs = self.params['epochs']
        csv_logger = self.get_csv_logger('training.csv')

        for i in range(nepochs):
            train_loss = self.train()
            test_results = self.test()
            test_results['train_loss'] = train_loss
            logger.info(f"Epoch {i}: " + ", ".join(f"{k} {v:.7f}" for k, v in test_results.items()))
            csv_logger.log(i, test_results)
            self.log_model_json(i, self.model)

        csv_logger.close()
        test_results = self.test()
        self.log_evaluation(test_results)

    def train(self):
        return self._train(self.train_dataloader, self.model, self.optimizer)

    def test(self):
        return self._test(self.test_dataloader, self.model)
