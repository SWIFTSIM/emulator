"""
Tests ancillary functions in the i/o submodule.
"""

import numpy as np

from swiftemulator import ModelValues
from swiftemulator.io.error import convert_dependent_error_to_standard_error


def test_dependent_error():
    model = ModelValues(
        {
            "0": {
                "independent": np.linspace(0, 1, 16),
                "dependent": np.random.rand(16),
            },
            "1": {
                "independent": np.linspace(0, 1, 16),
                "dependent": np.random.rand(16),
            },
        }
    )

    histogram = ModelValues(
        {
            "0": {
                "independent": np.linspace(0, 1, 16),
                "dependent": 10 * np.random.rand(16),
            },
            "1": {
                "independent": np.linspace(0, 1, 16),
                "dependent": 10 * np.random.rand(16),
            },
        }
    )

    no_log = convert_dependent_error_to_standard_error(
        scaling_relation=model,
        histogram=histogram,
        log_dependent=False,
        log_independent=False,
    )

    log = convert_dependent_error_to_standard_error(
        scaling_relation=model,
        histogram=histogram,
        log_dependent=True,
        log_independent=True,
    )

    return
