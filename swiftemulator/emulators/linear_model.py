"""
Linear Model Emulator
"""

import attr
import copy
import numpy as np
import sklearn.linear_model as lm

from typing import Hashable, List, Optional, Dict

from swiftemulator.backend.model_parameters import ModelParameters
from swiftemulator.backend.model_specification import ModelSpecification
from swiftemulator.backend.model_values import ModelValues


@attr.s
class LinearModelEmulator(object):
    """
    Generator for emulators for individual scaling relations.

    Parameters
    ----------

    model_specification: ModelSpecification
        Full instance of the model specification.

    model_parameters: ModelParameters
        Full instance of the model parameters.

    model_values: ModelValues
        Full instance of the model values describing
        this individual scaling relation.
    """

    model_specification: ModelSpecification = attr.ib(
        validator=attr.validators.instance_of(ModelSpecification)
    )
    model_parameters: ModelParameters = attr.ib(
        validator=attr.validators.instance_of(ModelParameters)
    )
    model_values: ModelValues = attr.ib(
        validator=attr.validators.instance_of(ModelValues)
    )

    ordering: Optional[List[Hashable]] = None
    parameter_order: Optional[List[str]] = None

    independent_variables: Optional[np.array] = None
    dependent_variables: Optional[np.array] = None
    dependent_variable_errors: Optional[np.array] = None

    emulator: Optional[lm.LinearRegression] = None

    def build_arrays(self):
        """
        Builds the arrays for passing to `george`.
        """

        model_values = self.model_values.model_values
        unique_identifiers = model_values.keys()
        number_of_independents = self.model_values.number_of_variables
        number_of_model_parameters = self.model_specification.number_of_parameters
        model_parameters = self.model_parameters.model_parameters

        independent_variables = np.empty(
            (number_of_independents, number_of_model_parameters + 1), dtype=np.float32
        )

        dependent_variables = np.empty((number_of_independents), dtype=np.float32)

        dependent_variable_errors = np.empty((number_of_independents), dtype=np.float32)

        self.parameter_order = self.model_specification.parameter_names
        self.ordering = []
        filled_lines = 0

        for unique_identifier in unique_identifiers:
            self.ordering.append(unique_identifier)

            # Unpack model parameters into an array
            model_parameter_array = np.array(
                [
                    model_parameters[unique_identifier][parameter]
                    for parameter in self.parameter_order
                ]
            )

            this_model = model_values[unique_identifier]
            model_independent = this_model["independent"]
            model_dependent = this_model["dependent"]
            model_error = this_model.get(
                "dependent_error", np.zeros(len(model_independent))
            )

            if np.ndim(model_error) != 1:
                raise AttributeError(
                    "Multiple dimensional errors are not currently supported in LM mode"
                )

            for line in range(len(model_independent)):
                independent_variables[filled_lines][0] = model_independent[line]
                independent_variables[filled_lines][1:] = model_parameter_array

                dependent_variables[filled_lines] = model_dependent[line]
                dependent_variable_errors[filled_lines] = model_error[line]

                filled_lines += 1

        assert filled_lines == number_of_independents

        self.independent_variables = independent_variables
        self.dependent_variables = dependent_variables
        self.dependent_variable_errors = dependent_variable_errors

    def fit_model(self, lasso_model_alpha=0.0):
        """
        Fits the GPE model.

        Parameters
        ----------

        lasso_model_alpha, float
            Alpha for the Lasso model. If this is 0.0 (the default)
            basic linear regression is used.
        """

        if self.independent_variables is None:
            self.build_arrays()

        if lasso_model_alpha == 0.0:
            linear_model = lm.LinearRegression(fit_intercept=True, normalize=True)
        else:
            linear_model = lm.Lasso(alpha=lasso_model_alpha)

        # Conform the model to the modelling protocol
        linear_model.fit(self.independent_variables, self.dependent_variables)

        self.emulator = linear_model

        return

    def predict_values(
        self, independent: np.array, model_parameters: Dict[str, float]
    ) -> np.array:
        """
        Predict values from the trained emulator contained within this object.

        Parameters
        ----------

        independent, np.array
            Independent continuous variables to evaluate the emulator
            at.

        model_parameters: Dict[str, float]
            The point in model parameter space to create predicted
            values at.

        Returns
        -------

        dependent_predictions, np.array
            Array of predictions, if the emulator is a function f, these
            are the predicted values of f(independent) evaluted at the position
            of the input model_parameters.

        dependent_prediction_errors, np.array
            Errors on the model predictions. For the linear model these are
            all zeroes, as the errors are unconstrained.
        """

        if self.emulator is None:
            raise AttributeError(
                "Please train the emulator with fit_model before attempting "
                "to make predictions."
            )

        model_parameter_array = np.array(
            [model_parameters[parameter] for parameter in self.parameter_order]
        )

        t = np.empty(
            (len(independent), len(model_parameter_array) + 1), dtype=np.float32
        )

        for line, value in enumerate(independent):
            t[line][0] = value
            t[line][1:] = model_parameter_array

        model = self.emulator.predict(X=t)

        return model, np.zeros_like(model)
