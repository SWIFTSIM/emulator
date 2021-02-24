"""
A basic linear model based on the scikit-learn multidimensional
mean model.
"""

import attr
import numpy as np
import sklearn.linear_model as lm

from .base import MeanModel

from typing import Optional, Union


@attr.s
class LinearMeanModel(MeanModel):
    """
    A linear mean model; fits a linear model to the
    multidimensional parameter space.

    Under the hood, this uses the ``sklearn.linear_model.lm``.

    Parameters
    ----------

    lasso_model_alpha: float, optional
        ``alpha`` for the Lasso model. If this is zero (the default)
        we fit a basic linear regression model is used for performance
        reasons.
    """

    lasso_model_alpha: float = attr.ib(default=0.0)

    model: Optional[Union[lm.LinearRegression, lm.Lasso]] = None

    def train(self, independent: np.ndarray, dependent: np.ndarray) -> None:
        """
        Train the model. See :class:`MeanModel` for more information.
        """

        self.model = (
            lm.LinearRegression(fit_intercept=True)
            if self.lasso_model_alpha == 0.0
            else lm.Lasso(alpha=self.lasso_model_alpha)
        )

        self.model.fit(independent, dependent)

        return

    def predict(self, independent: np.ndarray) -> np.ndarray:
        """
        Predict using the model. See :class:`MeanModel` for more information.
        """

        dependent = self.model.predict(independent)

        return dependent
