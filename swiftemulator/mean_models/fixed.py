"""
Fixed mean model. Basic, fixed, 0th order model.
"""

from .base import MeanModel

import numpy as np
import attr

from typing import Optional


@attr.s
class FixedMeanModel(MeanModel):
    """
    A basic offset mean model. Uses a fixed (manual) value for the mean
    model and does not allow it to be changed.
    """

    model: float = attr.ib(default=1.0)

    def train(self, independent: np.ndarray, dependent: np.ndarray) -> None:
        """
        Train the model. See :class:`MeanModel` for more information.
        """

        # This is a no-op function because the model is fixed.

        return

    def predict(self, independent: np.ndarray) -> np.ndarray:
        """
        Predict using the model. See :class:`MeanModel` for more information.
        """

        dependent = np.ones(independent.shape[0], dtype=independent.dtype) * self.model

        return dependent
