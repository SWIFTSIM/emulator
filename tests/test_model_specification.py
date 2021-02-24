"""
Tests for objects in ``swift-emulator/backend/model_specification.py``
"""

from swiftemulator.backend.model_specification import ModelSpecification
import pytest


def test_basic_creation():
    """
    Basic creation of the model specification.
    """

    model_spec = ModelSpecification(
        number_of_parameters=3,
        parameter_names=["Hello", "World", "Foo"],
        parameter_limits=[[0.0, 1.0], [0.0, 1.0], [0.2, 0.3]],
    )

    assert model_spec.parameter_names == model_spec.parameter_printable_names

    return


def test_failure():
    """
    Test the failure when parameter limits are not included.
    """

    with pytest.raises(AttributeError):
        _ = ModelSpecification(
            number_of_parameters=3,
            parameter_names=["Hello", "World", "Foo"],
            parameter_limits=[[0.0, 1.0], [0.0, 1.0]],
        )

    return
