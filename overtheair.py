"""Runs long experiments with the OverTheAirExperiment class, which implements
the most basic version of the over-the-air analog regime."""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021


from experiments import OverTheAirExperiment
from run_experiments import run_experiments

run_experiments(OverTheAirExperiment, __doc__)
