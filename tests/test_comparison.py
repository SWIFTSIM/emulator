"""
Tests the code used for comparisons between observational data and
emulated results.
"""

from swiftemulator.comparison import continuous_difference

import numpy as np


def test_continuous_difference():
    """
    Tests an easily known example.
    """

    independent_A = np.array([0.0, 1.0])
    independent_B = np.array([0.0, 2.0])

    dependent_A = np.array([0.0, 0.0])
    dependent_B = np.array([1.0, 1.0])

    expected_integral = 1.0

    numerical_integral = continuous_difference(
        independent_A=independent_A,
        dependent_A=dependent_A,
        independent_B=independent_B,
        dependent_B=dependent_B,
    )

    assert np.isclose(expected_integral, numerical_integral)


def test_continuous_difference_extrapolation():
    """
    Tests a case where we need to extrapolate A, and uses a
    cross-over.
    """
    independent_A = np.array([0.0, 1.0])
    independent_B = np.array([0.0, 1.0, 2.0])

    dependent_A = np.array([0.0, 1.0])
    dependent_B = np.array([1.0, 0.0, 0.0])

    expected_integral = 1.5

    numerical_integral = continuous_difference(
        independent_A=independent_A,
        dependent_A=dependent_A,
        independent_B=independent_B,
        dependent_B=dependent_B,
        difference_range=[0.0, 2.0],
    )

    assert np.isclose(expected_integral, numerical_integral)
