"""
Gaussian Process Emulator using an emulator for each bins
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
from swiftemulator.emulators.gaussian_process import GaussianProcessEmulator
from swiftemulator.mean_models import MeanModel

from scipy.optimize import minimize


@attr.s
class GaussianProcessEmulatorBins(BaseEmulator):
    """
    Generator for emulators for individual scaling relations.
    Uses a GP for each seperate bin.

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

    n_bins: int = None
    bin_model_values: List[Dict[str, np.array]] = None
    bin_centers: List[float] = None
    bin_gaussian_process: List[george.GP] = None

    def _build_arrays(
        self,
        model_specification: ModelSpecification,
        model_parameters: ModelParameters,
        model_values: ModelValues,
    ):
        """
        Builds the arrays for passing to `george`. As we aim to build
        a emulator for each dependent bin, this creates a dictionary containing
        the dependent, dependent errors and the variables beloning to each bin
        in a format the `george` likes. It also links this to the bin centers,
        which are stored as a separate dictionary.

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

        self.parameter_order = self.model_specification.parameter_names
        number_of_model_parameters = self.model_specification.number_of_parameters
        unique_identifiers = list(model_values.model_values.keys())

        bin_centers = np.unique(
            [
                item
                for uid in unique_identifiers
                for item in model_values.model_values[uid]["independent"]
            ]
        )

        self.n_bins = len(bin_centers)
        self.bin_centers = bin_centers

        bin_model_values = []

        for bin_center in bin_centers:
            bin_variables = []
            bin_dependent = []
            bin_dependent_errors = []

            for unique_identifier in unique_identifiers:
                uniq_model_values = model_values.model_values[unique_identifier]
                bin_independent_uniq = uniq_model_values["independent"]

                # Which one of our bins corresponds to the bin of interest?
                # bin_mask only has one (or zero) True values.
                bin_mask = bin_independent_uniq == bin_center

                if not bin_mask.any():
                    continue

                bin_dependent_uniq = uniq_model_values["dependent"][bin_mask][0]
                bin_dependent_errors_uniq = uniq_model_values.get(
                    "dependent_error", np.zeros(len(bin_independent_uniq))
                )[bin_mask][0]
                bin_independent_uniq = bin_independent_uniq[bin_mask][0]

                bin_variables_uniq = [
                    model_parameters.model_parameters[unique_identifier][parameter]
                    for parameter in self.parameter_order
                ]

                bin_dependent.append(bin_dependent_uniq)
                bin_dependent_errors.append(bin_dependent_errors_uniq)
                bin_variables.append(bin_variables_uniq)

                if np.ndim(bin_dependent_errors) != 1:
                    raise AttributeError(
                        "Multiple dimensional errors are not currently supported in GPE mode"
                    )

            bin_variables = np.array(bin_variables).reshape(
                (int(len(bin_dependent)), number_of_model_parameters)
            )

            bin_model_values.append(
                {
                    "independent": bin_variables,
                    "dependent": np.array(bin_dependent).flatten(),
                    "dependent_errors": np.array(bin_dependent_errors).flatten(),
                }
            )

        self.bin_model_values = bin_model_values

        return

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

        if self.bin_model_values is None:
            # Creates independent_variables, dependent_variables.
            self._build_arrays(
                model_specification=model_specification,
                model_parameters=model_parameters,
                model_values=model_values,
            )

        unique_bin_identifiers = list(range(self.n_bins))
        bin_model_values = self.bin_model_values

        bin_gaussian_process = []

        for bin_index in unique_bin_identifiers:
            independent_variables = bin_model_values[bin_index]["independent"]
            dependent_variables = bin_model_values[bin_index]["dependent"]
            dependent_variable_errors = bin_model_values[bin_index]["dependent_errors"]

            if self.kernel is None:
                number_of_kernel_dimensions = (
                    self.model_specification.number_of_parameters
                )

                self.kernel = 1 ** 2 * george.kernels.ExpSquaredKernel(
                    np.ones(number_of_kernel_dimensions),
                    ndim=number_of_kernel_dimensions,
                )

            if self.mean_model is not None:
                self.mean_model.train(
                    independent=independent_variables,
                    dependent=dependent_variables,
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

            bin_gaussian_process.append(gaussian_process)

        self.bin_gaussian_process = bin_gaussian_process

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
            at. If the emulator is discrete, these are only allowed to be
            the discrete independent variables that the emulator was trained at
            (disregarding the additional 'independent' model parameters, below).
            These can be found in this object in the ``bin_centers`` attribute.

        model_parameters: Dict[str, float]
            The point in model parameter space to create predicted
            values at.

        Returns
        -------

        dependent_predictions, np.array
            Array of predictions, if the emulator is a function f, these
            are the predicted values of f(independent) evaluted at the position
            of the input ``model_parameters``.

        dependent_prediction_errors, np.array
            Errors on the model predictions. For models where the errors are
            unconstrained, this is an array of zeroes.

        Raises
        ------

        AttributeError
            When the model has not been trained before trying to make a
            prediction, or when attempting to evaluate the model at
            disallowed independent variables.
        """

        if self.bin_gaussian_process is None:
            raise AttributeError(
                "Please train the emulator with fit_model before attempting "
                "to make predictions."
            )

        # First calculate which indices in bin_centers (and hence
        # bin_gaussian_processes) correspond to the requested ``independent``
        # variables.

        array_centers = np.array(self.bin_centers)
        gpe_ordering = []

        for requested_independent_variable in independent:
            try:
                gpe_ordering.append(
                    np.where(array_centers == requested_independent_variable)[0][0]
                )
            except IndexError:
                raise AttributeError(
                    f"Requested independent variable {independent} not valid, ",
                    f"this instance of GPE Bins is only valid at {array_centers}.",
                )

        model_parameter_array = np.array(
            [model_parameters[parameter] for parameter in self.parameter_order]
        )

        # George must predict a value for more than one point at a time, so
        # generate two fake points either side of the one of interest.
        model_parameter_array_sample = np.append(
            0.98 * model_parameter_array, model_parameter_array
        )
        model_parameter_array_sample = np.append(
            model_parameter_array_sample, 1.02 * model_parameter_array
        ).reshape(3, len(model_parameter_array))

        dependent_predictions = []
        dependent_prediction_errors = []

        for emulator_index in gpe_ordering:
            gp = self.bin_gaussian_process[emulator_index]
            model_values = self.bin_model_values[emulator_index]

            model, errors = gp.predict(
                y=model_values["dependent"],
                t=model_parameter_array_sample,
                return_cov=False,
                return_var=True,
            )

            # Remove fake points required to ensure george returns a prediction.
            dependent_predictions.append(model[1])
            dependent_prediction_errors.append(errors[1])

        return (
            np.array(dependent_predictions),
            np.array(dependent_prediction_errors),
        )
