"""
A base model to be overloaded in further implementations.
"""

from george.modeling import CallableModel
from copy import deepcopy

import numpy as np
import attr


@attr.s
class MeanModel(object):
    """
    A base mean model which describes the basic layout.
    Each function must be overloaded.
    """

    def train(self, independent: np.ndarray, dependent: np.ndarray) -> None:
        """
        Trains the mean model to predict dependent(independent).

        By convention, if there is an underlying model object,
        this should be stored in ``self.model``.

        Parameters
        ----------

        independent: np.ndarray
            Independent variables. Should be in the same format as is passed
            to ``george``.

        dependent: np.ndarray
            Dependent variables, to be predicted from the independent
            variables. Should be in the same format as is passed to  ``george``.

        Raises
        ------

        NotImplementedError
            If not implemented.
        """

        raise NotImplementedError

    def predict(self, independent: np.ndarray) -> np.ndarray:
        """
        Predicts dependent variables from independent variables,
        using the predictive model.

        Parameters
        ----------

        independent: np.ndarray
            Independent variables to predict dependent variables from,
            in the same format as is passed to ``george``.


        Returns
        -------

        dependent: np.ndarray
            Dependent variables that are in the same format that
            ``george`` expects.


        Raises
        ------

        NotImplementedError
            If not implemented

        AttributeError
            If the ``model`` is not trained.

        """

        raise NotImplementedError

    @property
    def george_model(self) -> CallableModel:
        """
        Get the ``george`` ``CallableModel`` instance that
        corresponds to this system. Returns a copy of the predict
        function associated with the current instance.
        """

        return CallableModel(self.copy().predict)

    def copy(self) -> "MeanModel":
        """
        Copy self to a new version of the model. Required should
        you wish to re-use a version of this object later on, as
        otherwise the ``model`` parameter will be mutated.
        """

        return deepcopy(self)
