"""Runs long experiments with the DynamicRangeFederatedExperiment class, which
implements an unconstrained communication scheme with some dynamic quantization
range adjustment. This is mainly intended for debugging/sanity checks against
DynamicRangeQuantizationFederatedExperiment."""

# Chuan-Zheng Lee <czlee@stanford.edu>
# August 2021


from experiments import DynamicRangeFederatedExperiment
from run_experiments import run_experiments

run_experiments(DynamicRangeFederatedExperiment, __doc__)
