"""Convenience functions for processing optimizer arguments.

The handling of optimizers is slightly different between the federated and
centralized cases, so this file just implements common functionality. The
primary purpose of these functions is to provide common handling of command-line
arguments, to specify things like optimizer class, learning rate and schedulers.

The federated experiments will implement each of these at the clients. The
centralized experiment will of course use one optimizer.
"""

import logging

from torch import optim

logger = logging.getLogger(__name__)

optimizer_choices = ['sgd', 'adam']


def make_optimizer(parameters, algorithm, lr, momentum, weight_decay):

    if algorithm == 'sgd':

        logger.debug(f"Optimizer: SGD with lr {lr}, momentum {momentum}, weight decay {weight_decay}")

        return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)

    elif algorithm == 'adam':

        if momentum != 0.0:
            logger.warning(f"Momentum is not used in the Adam optimizer, ignoring value: {momentum}")

        logger.debug(f"Optimizer: Adam with lr {lr}, weight decay {weight_decay}")

        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

    else:
        raise ValueError(f"Unrecognized optimizer algorithm option: {algorithm}")


def make_scheduler(spec, optimizer):
    """Parses a string specifying an LR scheduler."""

    if not spec:  # either empty string or None (or any False value)
        logger.debug("No LR scheduler")
        return None

    scheduler_type, parameters = spec.split("-", maxsplit=1)

    if scheduler_type == "multistep":

        if "-" in parameters:
            parameters, gamma = parameters.split("-")
            gamma = float(gamma)
        else:
            gamma = 0.1  # default

        milestones = [int(m) for m in parameters.split(",")]

        logger.debug(f"LR scheduler: multi-step with milestones {milestones}, gamma {gamma}")

        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    else:
        raise ValueError(f"Unrecognized scheduler option: {scheduler_type}")
