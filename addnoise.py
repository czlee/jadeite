"""Plain machine learning, but with noise added to the model after each training
round."""

# Chuan-Zheng Lee <czlee@stanford.edu>
# August 2021


from experiments import SimpleExperimentWithNoise
from run_experiments import run_experiments

run_experiments(SimpleExperimentWithNoise, __doc__)
