"""
Tests the emulator generator in ``swift-emulator/backend/emulator_generator.py``
"""

from swiftemulator.sensitivity.cross_check_bins import CrossCheckBins

from swiftemulator.backend.emulator_generator import (
    ModelParameters,
    ModelSpecification,
    ModelValues,
)

import numpy as np


def test_basic_emulator_generator():
    """
    Basic emulator generator test.
    """

    model_spec = ModelSpecification(
        number_of_parameters=2,
        parameter_names=["x", "y"],
        parameter_limits=[[1.0, 3.0], [0.0, 1.0]],
    )

    my_model_parameters = {
        0: {
            "x": 1,
            "y": 1,
        },
        1: {
            "x": 2,
            "y": 1,
        },
        2: {"x": 3, "y": 1},
    }

    model_parameters = ModelParameters(model_parameters=my_model_parameters)

    input_model_values = {
        0: {
            "independent": np.arange(10),
            "dependent": np.random.rand(10),
            "dependent_error": np.random.rand(10),
        },
        1: {
            "independent": np.arange(10),
            "dependent": np.random.rand(10),
            "dependent_error": np.random.rand(10),
        },
        2: {
            "independent": np.arange(9),
            "dependent": np.random.rand(9),
        },
    }

    model_values = ModelValues(model_values=input_model_values)

    cross_check_bins = CrossCheckBins()

    cross_check_bins.build_emulators(
        model_specification=model_spec,
        model_parameters=model_parameters,
        model_values=model_values,
    )
