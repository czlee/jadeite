"""Runs long experiments with the SimpleQuantizationFederatedExperiment class,
which implements a simple form of stochastic quantization for the digital
scheme.

(It can also do deterministic quantization, see the --rounding-method option.
The script name's a hangover from before this was implemented.)"""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021


from experiments import SimpleQuantizationFederatedExperiment
from run_experiments import run_experiments

run_experiments(SimpleQuantizationFederatedExperiment, __doc__)
