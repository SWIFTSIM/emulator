"""
Gaussian Process Emulator using an emulator for each bins
"""

import attr
import copy
import numpy as np
import george
import sklearn.linear_model as lm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from typing import Hashable, List, Optional, Dict

from swiftemulator.backend.model_parameters import ModelParameters
from swiftemulator.backend.model_specification import ModelSpecification
from swiftemulator.backend.model_values import ModelValues
from swiftemulator.emulators.gaussian_process import GaussianProcessEmulator

from scipy.optimize import minimize


@attr.s
class GaussianProcessEmulatorBins(object):
    """
    Generator for emulators for individual scaling relations.
    Uses a GP for each seperate bin.

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

    n_bins: int = None
    bin_model_values: Dict[Hashable, Dict[str, np.array]] = None
    bin_centers: Dict[Hashable, float] = None
    bin_gaussian_process: Dict[Hashable, george.GP] = None

    def build_arrays(self):
        """
        Builds the arrays for passing to `george`.
        """

        model_values = self.model_values.model_values
        model_parameters = self.model_parameters.model_parameters
        self.parameter_order = self.model_specification.parameter_names
        number_of_model_parameters = self.model_specification.number_of_parameters
        unique_identifiers = self.model_values.model_values.keys()

        independent_values = np.array([])
        for unique_identifier in unique_identifiers:
            independent_values = np.append(
                independent_values, model_values[unique_identifier]["independent"])

        bin_centers_array = np.unique(independent_values)
        self.n_bins = len(bin_centers_array)

        bin_centers = {}
        for index in range(self.n_bins):
            bin_centers[index] = bin_centers_array[index]

        self.bin_centers = bin_centers

        bin_model_values = {}
        for index in bin_centers.keys():
            bin_variables = np.array([])
            bin_dependent = np.array([])
            bin_dependent_errors = np.array([])
            for unique_identifier in unique_identifiers:
                bin_independent_uniq = model_values[unique_identifier]["independent"]
                if len(bin_independent_uniq[bin_independent_uniq == bin_centers[index]]) != 0:
                    bin_dependent_uniq = model_values[unique_identifier][
                        "dependent"][bin_independent_uniq == bin_centers[index]][0]
                    bin_dependent_errors_uniq = model_values[unique_identifier].get(
                        "dependent_error", np.zeros(len(bin_independent_uniq)))[bin_independent_uniq == bin_centers[index]][0]
                    bin_independent_uniq = bin_independent_uniq[bin_independent_uniq ==
                                                                bin_centers[index]][0]
                    bin_variables_uniq = np.array(
                        [
                            model_parameters[unique_identifier][parameter]
                            for parameter in self.parameter_order
                        ]
                    )
                    bin_dependent = np.append(
                        bin_dependent, bin_dependent_uniq)
                    bin_dependent_errors = np.append(
                        bin_dependent_errors, bin_dependent_errors_uniq)
                    bin_variables = np.append(
                        bin_variables, bin_variables_uniq)

                    if np.ndim(bin_dependent_errors) != 1:
                        raise AttributeError(
                            "Multiple dimensional errors are not currently supported in GPE mode"
                        )

            bin_variables = bin_variables.reshape(
                (int(len(bin_dependent)), number_of_model_parameters))
            variabledict = {"independent": bin_variables,
                            "dependent": bin_dependent, "dependent_errors": bin_dependent_errors}
            bin_model_values[index] = variabledict

        self.bin_model_values = bin_model_values

        return

    def fit_model(
        self,
        kernel=None,
        fit_model: str = "none",
        lasso_model_alpha: float = 0.0,
        polynomial_degree: int = 1,
    ):
        """
        Fits the GPE model to each bin seperatly.

        Parameters
        ----------

        kernel, george.kernels
            The ``george`` kernel to use. The GPE here uses a copy
            of this instance. By default, this is the
            ``ExpSquaredKernel`` in George

        fit_model, str
            Type of model to use for mean fitting, Optional, defaults
            to none which is a pure GP modelling. Options: "linear" and
            "polynomial"

        lasso_model_alpha, float
            Alpha for the Lasso model (only used of course when asking to
            ``fit_linear_model``). If this is 0.0 (the default) basic linear
            regression is used.

        polynomial_degree, int
            Maximal degree of the polynomial surface, default 1; linear for each
            parameter
        """

        unique_bin_identifiers = self.bin_model_values.keys()
        bin_model_values = self.bin_model_values

        if self.bin_model_values is None:
            self.build_arrays()

        bin_gaussian_process = {}
        for unique_identifier in unique_bin_identifiers:
            independent_variables = bin_model_values[unique_identifier]["independent"]
            dependent_variables = bin_model_values[unique_identifier]["dependent"]
            dependent_variable_errors = bin_model_values[unique_identifier]["dependent_errors"]

            number_of_kernel_dimensions = (
                self.model_specification.number_of_parameters
            )

            kernel = 1 ** 2 * george.kernels.ExpSquaredKernel(
                np.ones(number_of_kernel_dimensions), ndim=number_of_kernel_dimensions
            )

            if fit_model == "linear":
                if lasso_model_alpha == 0.0:
                    linear_model = lm.LinearRegression(fit_intercept=True)
                else:
                    linear_model = lm.Lasso(alpha=lasso_model_alpha)

                # Conform the model to the modelling protocol
                linear_model.fit(independent_variables,
                                 dependent_variables)
                linear_mean = george.modeling.CallableModel(
                    function=linear_model.predict)

                gaussian_process = george.GP(
                    copy.copy(kernel),
                    fit_kernel=True,
                    mean=linear_mean,
                    fit_mean=False,
                )
            elif fit_model == "polynomial":
                polynomial_model = Pipeline(
                    [
                        ("poly", PolynomialFeatures(degree=polynomial_degree)),
                        ("linear", lm.LinearRegression(fit_intercept=True)),
                    ]
                )

                # Conform the model to the modelling protocol
                polynomial_model.fit(
                    independent_variables, dependent_variables)
                linear_mean = george.modeling.CallableModel(
                    function=polynomial_model.predict
                )

                gaussian_process = george.GP(
                    copy.copy(kernel),
                    fit_kernel=True,
                    mean=linear_mean,
                    fit_mean=False,
                )
            else:
                if fit_model != "none":
                    raise ValueError(
                        "Your choice of fit_model is currently not supported."
                    )

                gaussian_process = george.GP(copy.copy(kernel))

            # TODO: Figure out how to include non-symmetric errors.
            gaussian_process.compute(
                x=independent_variables,
                yerr=dependent_variable_errors,
            )

            def negative_log_likelihood(p):
                gaussian_process.set_parameter_vector(p)
                return -gaussian_process.log_likelihood(dependent_variables)

            def grad_negative_log_likelihood(p):
                gaussian_process.set_parameter_vector(p)
                return -gaussian_process.grad_log_likelihood(dependent_variables)

            # Optimize the hyperparameter values in the emulator
            result = minimize(
                fun=negative_log_likelihood,
                x0=gaussian_process.get_parameter_vector(),
                jac=grad_negative_log_likelihood,
            )

            # Load in the optimal hyperparameters
            gaussian_process.set_parameter_vector(result.x)

            bin_gaussian_process[unique_identifier] = gaussian_process

        self.bin_gaussian_process = bin_gaussian_process

        return

    def predict_values(self, model_parameters: Dict[str, float]
                       ) -> np.array:
        """
        Predict values from the trained emulator contained within this object.

        Parameters
        ----------

        model_parameters: Dict[str, float]
            The point in model parameter space to create predicted
            values at.

        Returns
        -------

        independent_array, np.array
            array with the x_values corresponding to the dependent
            values

        dependent_predictions, np.array
            Array of predictions, if the emulator is a function f, these
            are the predicted values of f(independent) evaluted at the position
            of the input model_parameters.

        dependent_prediction_errors, np.array
            Variance on the model predictions.
        """

        unique_bin_identifiers = self.bin_model_values.keys()
        bin_centers = self.bin_centers
        bin_model_values = self.bin_model_values

        if self.bin_gaussian_process is None:
            raise AttributeError(
                "Please train the emulator with fit_model before attempting "
                "to make predictions."
            )

        model_parameter_array = np.array(
            [model_parameters[parameter]
                for parameter in self.parameter_order]
        )

        model_parameter_array_sample = np.append(
            0.98*model_parameter_array, model_parameter_array)
        model_parameter_array_sample = np.append(
            model_parameter_array_sample, 1.02 * model_parameter_array).reshape(3, len(model_parameter_array))

        x_array = np.empty(len(unique_bin_identifiers), dtype=np.float32)
        y_array = np.empty(len(unique_bin_identifiers), dtype=np.float32)
        y_error_array = np.empty(len(unique_bin_identifiers), dtype=np.float32)

        for index, unique_identifier in enumerate(unique_bin_identifiers):

            x_array[index] = bin_centers[unique_identifier]

            y = bin_model_values[unique_identifier]["dependent"]

            uniq_gp = self.bin_gaussian_process[unique_identifier]

            model, errors = uniq_gp.predict(
                y=y, t=model_parameter_array_sample, return_cov=False, return_var=True
            )

            y_array[index] = model[1]
            y_error_array[index] = errors[1]

        return x_array, y_array, y_error_array
