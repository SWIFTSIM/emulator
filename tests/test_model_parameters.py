"""
Tests for objects in ``swift-emulator/backend/model_specification.py``
"""

from swiftemulator.backend.model_parameters import ModelParameters
import pytest


def test_basic_model_parameters():
    """
    A basic test with a valid set of model parameters.
    """

    my_model_parameters = {
        0: {
            "x": 1,
            "y": 1,
        },
        1: {
            "x": 2,
            "y": 1,
        },
    }

    _ = ModelParameters(model_parameters=my_model_parameters)

    return


def test_basic_model_parameters_raise():
    """
    A basic test with an invalid set of model parameters.
    """

    my_model_parameters = {
        0: {
            "x": 1,
            "y": 1,
        },
        1: {
            "x": 2,
            "z": 1,
        },
    }

    with pytest.raises(AttributeError):
        _ = ModelParameters(model_parameters=my_model_parameters)

    return


def test_find_closest_model():
    """
    A simple test to see if it finds the closest model
    """

    model_point = {"x": 0.4, "y": 0.9}

    my_model_parameters = {
        0: {
            "x": 1,
            "y": 1,
        },
        1: {
            "x": 2,
            "y": 1,
        },
        2: {
            "x": 0.45,
            "y": 0.91,
        },
    }

    my_model_parameters = ModelParameters(model_parameters=my_model_parameters)
    model, _ = my_model_parameters.find_closest_model(model_point)

    assert model[0] == 2

    model, _ = my_model_parameters.find_closest_model(
        model_point, number_of_close_models=2
    )

    assert model[0] == 2

    return
