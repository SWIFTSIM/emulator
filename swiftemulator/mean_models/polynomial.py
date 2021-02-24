"""
A basic polynomial model based on the scikit-learn multidimensional
mean model.
"""

import attr
import numpy as np
import sklearn.linear_model as lm

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from .base import MeanModel

from typing import Optional, Union


@attr.s
class PolynomialMeanModel(MeanModel):
    """
    A polynomial mean model; fits a linear model to the
    multidimensional parameter space.

    Under the hood, this uses the ``sklearn.linear_model.lm``.

    Parameters
    ----------

    degree: int, optional
        Maximal degree of the polynomial surface, default 1 (linear
        in each parameter).
    """

    degree: int = attr.ib(default=1)

    model: Optional[Union[lm.LinearRegression, lm.Lasso]] = None

    def train(self, independent: np.ndarray, dependent: np.ndarray) -> None:
        """
        Train the model. See :class:`MeanModel` for more information.
        """

        self.model = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=self.degree)),
                ("linear", lm.LinearRegression(fit_intercept=True)),
            ]
        )

        self.model.fit(independent, dependent)

        return

    def predict(self, independent: np.ndarray) -> np.ndarray:
        """
        Predict using the model. See :class:`MeanModel` for more information.
        """

        dependent = self.model.predict(independent)

        return dependent
