"""Runs long experiments with the DynamicRangeQuantizationFederatedExperiment
class, which implements a stochastic quantization for the digital scheme with
some dynamic quantization range adjustment."""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021


from experiments import DynamicRangeQuantizationFederatedExperiment
from run_experiments import run_experiments

run_experiments(DynamicRangeQuantizationFederatedExperiment, __doc__)
