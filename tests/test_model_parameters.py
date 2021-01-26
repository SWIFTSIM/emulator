"""
Tests for objects in ``swift-emulator/backend/model_specification.py``
"""

from swiftemulator.backend.model_parameters import ModelParmeters
import pytest


def test_basic_model_parameters():
    """
    A basic test with a valid set of model parameters.
    """

    my_model_parameters = {0: {"x": 1, "y": 1,}, 1: {"x": 2, "y": 1,}}

    _ = ModelParmeters(model_parameters=my_model_parameters)

    return


def test_basic_model_parameters_raise():
    """
    A basic test with an invalid set of model parameters.
    """

    my_model_parameters = {0: {"x": 1, "y": 1,}, 1: {"x": 2, "z": 1,}}

    with pytest.raises(AttributeError):
        _ = ModelParmeters(model_parameters=my_model_parameters)

    return
