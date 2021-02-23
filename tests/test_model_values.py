"""
Tests for objects in ``swift-emulator/backend/model_values.py``
"""

from swiftemulator.backend.model_values import ModelValues
import pytest

import numpy as np


def test_basic_model_values():
    """
    A basic test of model values.
    """

    input_model_values = {
        0: {
            "independent": np.arange(10),
            "dependent": np.random.rand(10),
            "dependent_error": np.random.rand(10),
        },
        1: {
            "independent": np.arange(10),
            "dependent": np.random.rand(10),
            "dependent_error": np.random.rand(20).reshape(10, 2),
        },
        2: {
            "independent": np.arange(10),
            "dependent": np.random.rand(10),
        },
    }

    output_model = ModelValues(model_values=input_model_values)

    assert output_model.number_of_variables == 30


def test_basic_model_values_raise():
    """
    A basic test of model values, with incorrect data
    that raises an exception.
    """

    input_model_values = {
        0: {
            "independent": np.arange(10),
            "dependent": np.random.rand(10),
            "dependent_error": np.random.rand(10),
        },
        1: {
            "independent": np.arange(10),
            "dependent": np.random.rand(10),
            "dependent_error": np.random.rand(18).reshape(9, 2),
        },
        2: {
            "independent": np.arange(10),
            "dependent": np.random.rand(10),
        },
    }

    with pytest.raises(AttributeError):
        ModelValues(model_values=input_model_values)
