"""Class for a single experiment."""

import torch


class Experiment:
    """Class for a single experiment, by which we mean a single attempt to train
    a model."""

    default_params = {
        'epochs': 20,
        'batch_size': 64,
    }

    def __init__(
            self,
            train_dataset: torch.utils.data.Dataset,
            model: torch.nn.Module,
            loss_fn,
            optimizer: torch.optim.Optimizer,
            device='cpu',
            **params):

        self.train_dataset = train_dataset
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def train_once(self):
        """Trains through one epoch."""
        self.model.train()


class OverTheAirExperiment(Experiment):

    default_params = Experiment.default_params.copy()
    default_params.update({
        'rounds': 20,
        'epochs': 1,
        'clients': 10,
        'lr_client': 1e-2,
        'noise': 1.0,
        'power': 1.0,
        'parameter_radius': 1.0,
    })
