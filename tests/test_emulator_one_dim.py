"""
Tests the emulator in ``swift-emulator/emulators/gaussian_process_one_dim.py``
"""

from swiftemulator.backend.emulator_generator import (
    ModelParameters,
    ModelSpecification,
    ModelValues,
)

from swiftemulator.emulators.gaussian_process_one_dim import GaussianProcessEmulator1D

import numpy as np


def test_basic_gpe_with_1d():
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
            "independent": [None],
            "dependent": np.random.rand(1),
            "dependent_error": np.random.rand(1),
        },
        1: {
            "independent": [None],
            "dependent": np.random.rand(1),
            "dependent_error": np.random.rand(1),
        },
        2: {
            "independent": [None],
            "dependent": np.random.rand(1),
            "dependent_error": np.random.rand(1),
        },
    }

    model_values = ModelValues(model_values=input_model_values)

    gpe = GaussianProcessEmulator1D()

    gpe.fit_model(
        model_specification=model_spec,
        model_parameters=model_parameters,
        model_values=model_values,
    )
