"""Module containing experiment classes."""

# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021
# flake8: noqa

from .digital import SimpleQuantizationFederatedExperiment
from .experiment import SimpleExperiment
from .federated import FederatedAveragingExperiment
from .analog import OverTheAirExperiment, DynamicPowerOverTheAirExperiment
