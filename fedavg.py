"""Runs long experiments with the FederatedAveragingExperiment class,
which implements federated averaging with unconstrained communication."""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021


from experiments import FederatedAveragingExperiment
from run_experiments import run_experiments

run_experiments(FederatedAveragingExperiment, __doc__)
