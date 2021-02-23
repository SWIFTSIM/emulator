"""
Offset mean model. Basic 0th order model.
"""

from .base import MeanModel

import numpy as np
import attr

from typing import Optional


@attr.s
class OffsetMeanModel(MeanModel):
    """
    A basic offset mean model. Simply takes the mean
    of all of the dependent variables. Not likely to
    be useful in practice, but more of an example
    of using the protocol.
    """

    model: Optional[float] = None

    def train(self, independent: np.ndarray, dependent: np.ndarray) -> None:
        """
        Train the model. See :class:`MeanModel` for more information.
        """

        self.model = np.mean(dependent)

        return

    def predict(self, independent: np.ndarray) -> np.ndarray:
        """
        Predict using the model. See :class:`MeanModel` for more information.
        """

        dependent = np.ones(independent.shape[0], dtype=independent.dtype) * self.model

        return dependent
