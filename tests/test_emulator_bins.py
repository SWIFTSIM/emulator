"""
Tests the emulator generator in ``swift-emulator/backend/emulator_generator.py``
"""

from swiftemulator.backend.emulator_generator import (
    ModelParameters,
    ModelSpecification,
    ModelValues,
)

from swiftemulator.emulators.gaussian_process_bins import GaussianProcessEmulatorBins

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
            "independent": np.arange(10),
            "dependent": np.random.rand(10),
        },
    }

    model_values = ModelValues(model_values=input_model_values)

    gpe = GaussianProcessEmulatorBins()

    gpe.fit_model(
        model_specification=model_spec,
        model_parameters=model_parameters,
        model_values=model_values,
    )

    gpe.predict_values([0, 7], {"x": 1.5, "y": 1.0})
    gpe.predict_values_no_error([0, 7], {"x": 1.5, "y": 1.0})
