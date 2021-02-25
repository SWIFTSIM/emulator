"""
Tests model design code.
"""

from swiftemulator.backend.model_specification import ModelSpecification

from swiftemulator.design.latin import create_hypercube
from swiftemulator.design.random import create_cube


def test_create_hypercube():
    """
    Creates a hypercube and tests that values fall within the correct limits.

    Statistics testing of the hypercube is left to the individual user.
    """

    model_spec = ModelSpecification(
        number_of_parameters=4,
        parameter_names=["Hello", "World", "Foo", "Bar"],
        parameter_limits=[[-2.0, -1.0], [0.0, 1.0], [0.2, 0.3], [-1023.0, 1048.0]],
    )

    hypercube = create_hypercube(
        model_specification=model_spec,
        number_of_samples=120,
    )

    assert len(hypercube.model_parameters) == 120

    for model in hypercube.model_parameters.values():
        for parameter, limits in zip(
            model_spec.parameter_names, model_spec.parameter_limits
        ):
            sample = model[parameter]
            assert sample >= limits[0] and sample <= limits[1]

    return


def test_create_cube():
    """
    Creates a hypercube and tests that values fall within the correct limits.
    This time, we use the completely random design.

    Statistics testing of the hypercube is left to the individual user.
    """

    model_spec = ModelSpecification(
        number_of_parameters=4,
        parameter_names=["Hello", "World", "Foo", "Bar"],
        parameter_limits=[[-2.0, -1.0], [0.0, 1.0], [0.2, 0.3], [-1023.0, 1048.0]],
    )

    hypercube = create_cube(
        model_specification=model_spec,
        number_of_samples=120,
    )

    assert len(hypercube.model_parameters) == 120

    for model in hypercube.model_parameters.values():
        for parameter, limits in zip(
            model_spec.parameter_names, model_spec.parameter_limits
        ):
            sample = model[parameter]
            assert sample >= limits[0] and sample <= limits[1]

    return
