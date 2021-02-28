"""
Gaussian Process Emulator
"""

import attr
import copy
import numpy as np
import george

from typing import Hashable, List, Optional, Dict

from swiftemulator.emulators.base import BaseEmulator

from swiftemulator.backend.model_parameters import ModelParameters
from swiftemulator.backend.model_specification import ModelSpecification
from swiftemulator.backend.model_values import ModelValues
from swiftemulator.mean_models import MeanModel

from scipy.optimize import minimize


@attr.s
class GaussianProcessEmulator(BaseEmulator):
    """
    Generator for emulators for individual scaling relations.

    Parameters
    ----------

    kernel, george.kernels.Kernel, optional
        The ``george`` kernel to use. The GPE here uses a copy
        of this instance. By default, this is the
        ``ExpSquaredKernel`` in George

    mean_model, MeanModel, optional
        A mean model conforming to the ``swiftemulator`` mean model
        protocol (several pre-made models are available in the
        :mod:`swiftemulator.mean_models` module).
    """

    kernel: Optional[george.kernels.Kernel] = attr.ib(default=None)
    mean_model: Optional[MeanModel] = attr.ib(default=None)

    model_specification: Optional[ModelSpecification] = None
    model_parameters: Optional[ModelParameters] = None
    model_values: Optional[ModelValues] = None

    ordering: Optional[List[Hashable]] = None
    parameter_order: Optional[List[str]] = None

    independent_variables: Optional[np.array] = None
    dependent_variables: Optional[np.array] = None
    dependent_variable_errors: Optional[np.array] = None

    emulator: Optional[george.GP] = None

    def _build_arrays(
        self,
        model_specification: ModelSpecification,
        model_parameters: ModelParameters,
        model_values: ModelValues,
    ):
        """
        Builds the arrays for passing to `george`.

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

        self.model_specification = model_specification
        self.model_parameters = model_parameters
        self.model_values = model_values

        unique_identifiers = model_values.model_values.keys()
        number_of_independents = model_values.number_of_variables
        number_of_model_parameters = model_specification.number_of_parameters
        model_parameters = model_parameters.model_parameters

        independent_variables = np.empty(
            (number_of_independents, number_of_model_parameters + 1), dtype=np.float32
        )

        dependent_variables = np.empty((number_of_independents), dtype=np.float32)

        dependent_variable_errors = np.empty((number_of_independents), dtype=np.float32)

        self.parameter_order = model_specification.parameter_names
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

            this_model = model_values.model_values[unique_identifier]
            model_independent = this_model["independent"]
            model_dependent = this_model["dependent"]
            model_error = this_model.get(
                "dependent_error", np.zeros(len(model_independent))
            )

            if np.ndim(model_error) != 1:
                raise AttributeError(
                    "Multiple dimensional errors are not currently supported in GPE mode"
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

    def fit_model(
        self,
        model_specification: ModelSpecification,
        model_parameters: ModelParameters,
        model_values: ModelValues,
    ):
        """
        Fits the gaussian process model, as determined by the
        initialiser variables of the class (i.e. the kernel and
        the mean model).

        Parameters
        ----------

        model_specification: ModelSpecification
            Full instance of the model specification.

        model_parameters: ModelParameters
            Full instance of the model parameters.

        model_values: ModelValues
            Full instance of the model values describing
            this individual scaling relation.

        Notes
        -----

        This method uses copies of the internal kernel and mean model
        objects, as those objects contain slightly unhelpful state information.
        """

        if self.independent_variables is None:
            # Creates independent_variables, dependent_variables.
            self._build_arrays(
                model_specification=model_specification,
                model_parameters=model_parameters,
                model_values=model_values,
            )

        if self.kernel is None:
            number_of_kernel_dimensions = model_specification.number_of_parameters + 1

            self.kernel = 1 ** 2 * george.kernels.ExpSquaredKernel(
                np.ones(number_of_kernel_dimensions), ndim=number_of_kernel_dimensions
            )

        if self.mean_model is not None:
            self.mean_model.train(
                independent=self.independent_variables,
                dependent=self.dependent_variables,
            )

            gaussian_process = george.GP(
                copy.deepcopy(self.kernel),
                fit_kernel=True,
                mean=self.mean_model.george_model,
                fit_mean=False,
            )
        else:
            gaussian_process = george.GP(
                copy.deepcopy(self.kernel),
            )

        # TODO: Figure out how to include non-symmetric errors.
        gaussian_process.compute(
            x=self.independent_variables,
            yerr=self.dependent_variable_errors,
        )

        def negative_log_likelihood(p):
            gaussian_process.set_parameter_vector(p)
            return -gaussian_process.log_likelihood(self.dependent_variables)

        def grad_negative_log_likelihood(p):
            gaussian_process.set_parameter_vector(p)
            return -gaussian_process.grad_log_likelihood(self.dependent_variables)

        # Optimize the hyperparameter values in the emulator
        result = minimize(
            fun=negative_log_likelihood,
            x0=gaussian_process.get_parameter_vector(),
            jac=grad_negative_log_likelihood,
        )

        # Load in the optimal hyperparameters
        gaussian_process.set_parameter_vector(result.x)

        self.emulator = gaussian_process

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
            Errors on the model predictions.
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

        model, errors = self.emulator.predict(
            y=self.dependent_variables, t=t, return_cov=False, return_var=True
        )

        return model, errors
