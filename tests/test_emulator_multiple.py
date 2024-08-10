"""
Tests the emulator in multi_guassian_process.
"""

from swiftemulator.backend.emulator_generator import (
    ModelParameters,
    ModelSpecification,
    ModelValues,
)

from swiftemulator.emulators.multi_gaussian_process import (
    MultipleGaussianProcessEmulator,
)

import numpy as np


def test_basic_emulator_generator():
    """
    Basic emulator generator test. This should fall back to a single
    emulator.
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

    gpe = MultipleGaussianProcessEmulator()

    gpe.fit_model(
        model_specification=model_spec,
        model_parameters=model_parameters,
        model_values=model_values,
    )


def test_basic_emulator_generator_multiple():
    """
    Basic emulator generator test, with multiple trained emulators.
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

    gpe = MultipleGaussianProcessEmulator(independent_regions=[[None, 6], [4, None]])

    gpe.fit_model(
        model_specification=model_spec,
        model_parameters=model_parameters,
        model_values=model_values,
    )

    gpe.predict_values(np.array([0.2, 0.9, 9.9, 5.0]), {"x": 0.5, "y": 1})
    gpe.predict_values_no_error(np.array([0.2, 0.9, 9.9, 5.0]), {"x": 0.5, "y": 1})
