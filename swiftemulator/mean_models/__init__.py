"""
Mean models, usually used to create a basic parametrisation
of the space to enable the GPE to more efficiently model residuals.
"""

from swiftemulator.mean_models.base import MeanModel
from swiftemulator.mean_models.linear import LinearMeanModel
from swiftemulator.mean_models.offset import OffsetMeanModel
from swiftemulator.mean_models.polynomial import PolynomialMeanModel
