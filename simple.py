"""Just a plain machine learning, nothing federated about it."""

# Chuan-Zheng Lee <czlee@stanford.edu>
# August 2021


from experiments import SimpleExperiment
from run_experiments import run_experiments

run_experiments(SimpleExperiment, __doc__)
