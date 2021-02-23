"""
Tests the plotting and sensitivity analysis functions.
"""

from swiftemulator.sensitivity.basic import (
    binwise_sensitivity,
    plot_binwise_sensitivity,
)
from swiftemulator.backend.emulator_generator import (
    ModelParameters,
    ModelSpecification,
    ModelValues,
)

from matplotlib.pyplot import close

import numpy as np


def test_basic_binwise_sens():
    """
    Basic binwise sensitivity test; does not confirm functionality,
    as that would be too difficult in this scenario. This function
    just tests that given a set-up, we can construct and plot
    a sensitivity analysis without crashing.
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

    binwise = binwise_sensitivity(
        specification=model_spec,
        parameters=model_parameters,
        values=model_values,
    )

    fig, ax = plot_binwise_sensitivity(specification=model_spec, sensitivities=binwise)

    close(fig)
