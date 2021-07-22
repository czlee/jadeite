"""Runs long experiments with the DynamicPowerOverTheAirExperiment class,
which implements a simple form of dynamic power scaling within the analog
scheme."""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021


from experiments import DynamicPowerOverTheAirExperiment
from run_experiments import run_experiments

run_experiments(DynamicPowerOverTheAirExperiment, __doc__)
