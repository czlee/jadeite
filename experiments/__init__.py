"""Module containing experiment classes."""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

from .analog import DynamicPowerOverTheAirExperiment, OverTheAirExperiment
from .digital import (
    DynamicRangeFederatedExperiment,
    DynamicRangeQuantizationFederatedExperiment,
    SimpleQuantizationFederatedExperiment,
)
from .experiment import SimpleExperiment
from .federated import FederatedAveragingExperiment
from .simple_variants import SimpleExperimentWithNoise


experiments_by_name = {
    # basic experiments
    'simple': SimpleExperiment,
    'fedavg': FederatedAveragingExperiment,

    # analog scheme
    'overtheair': OverTheAirExperiment,
    'dynpower': DynamicPowerOverTheAirExperiment,

    # digital scheme
    'stocquant': SimpleQuantizationFederatedExperiment,
    'dynquant': DynamicRangeQuantizationFederatedExperiment,
    'dynrange': DynamicRangeFederatedExperiment,

    # other experiments
    'addnoise': SimpleExperimentWithNoise,
}
