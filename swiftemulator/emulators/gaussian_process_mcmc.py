"""
Gaussian Process Emulator
"""

import attr
import copy
import numpy as np
import george
import corner
import emcee
import matplotlib.pyplot as plt

from typing import Hashable, List, Optional, Dict

from swiftemulator.emulators.base import BaseEmulator

from swiftemulator.backend.model_parameters import ModelParameters
from swiftemulator.backend.model_specification import ModelSpecification
from swiftemulator.backend.model_values import ModelValues
from swiftemulator.mean_models import MeanModel

from scipy.optimize import minimize


@attr.s
class GaussianProcessEmulatorMCMC(BaseEmulator):
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

    burn_in_steps, int, optional
        Optional: Number of steps used for the burn-in part of the MCMC chain.
        Defaults to 50 for small intial test.

    mcmc_steps, int, optional
        Optional: Number of steps used for sampling the likelihood by the MCMC.
        chain. Defaults to 100 for small initial tests.

    walkers, int, optional
        Optional: Number of walkers used by the MCMC. Defaults to 40. Should
        (statistically) be at least 2 times the number of free parameters


    use_hyperparameter_error, bool, optional
        Switch for including errors originating from uncertain
        hyperparameters in the prediction outputs, (defaults to ``False``).

    samples_for_error, int, optional
        Number of MCMC samples to use for hyperparameter error estimation
        if ``use_hyperparameter_error`` is ``True``, defaults to 100.
    """

    kernel: Optional[george.kernels.Kernel] = attr.ib(default=None)
    mean_model: Optional[MeanModel] = attr.ib(default=None)
    burn_in_steps: int = attr.ib(default=50)
    mcmc_steps: int = attr.ib(default=100)
    walkers: int = attr.ib(default=40)
    use_hyperparameter_error: bool = attr.ib(default=False)
    samples_for_error: int = attr.ib(default=100)

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
            number_of_kernel_dimensions = (
                self.model_specification.number_of_parameters + 1
            )

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

        def log_likelihood(p):
            return -negative_log_likelihood(p)

        # Use scipy to find starting point to increase MCMC performance
        p0_start = minimize(
            fun=negative_log_likelihood,
            x0=gaussian_process.get_parameter_vector(),
            jac=grad_negative_log_likelihood,
        ).x

        # set up a MCMC sampling routine
        ndim = len(gaussian_process)
        sampler = emcee.EnsembleSampler(self.walkers, ndim, log_likelihood)
        # Assign starting points to the walkers, at a small distance from middle point
        p0 = p0_start + 1e-4 * np.random.randn(self.walkers, ndim)

        p0, _, _ = sampler.run_mcmc(p0, self.burn_in_steps)
        sampler.run_mcmc(p0, self.mcmc_steps)

        samples = sampler.get_chain()[:, self.burn_in_steps :, :].reshape((-1, ndim))

        result = np.mean(samples, axis=0)

        # Save the samples and best fit for error analysis
        self.hyperparameter_samples = samples
        self.hyperparameter_best_fit = result

        # Load in the optimal hyperparameters
        gaussian_process.set_parameter_vector(result)

        self.emulator = gaussian_process

        return

    def plot_hyperparameter_distribution(self, filename=None, labels=None):
        """
        Makes a cornerplot of the MCMC samples obtained when fitting the model

        Parameters
        ----------

        filename, None, str
            Name for the file to which the plot is saved. Optional, if None it
            will show the image.

        labels, None, list[Hashable]
            labels to add to the different plots. Optional, if None it will
            take the kernel names

        Note
        ----

        By using this function you solemnly swear to never try to infer anything
        from the hyperparameters, except whether they are converged.

        """

        if self.hyperparameter_samples is None:
            raise AttributeError(
                "Please train the emulator with fit_model before attempting "
                "to look at the hyperparameters."
            )

        if labels is None:
            labels = self.emulator.get_parameter_names()

        corner.corner(
            self.hyperparameter_samples,
            labels=labels,
            bins=40,
            hist_bin_factor=10,
            quantiles=[0.16, 0.5, 0.84],
            levels=(0.68, 0.95),
            color="k",
            smooth1d=1,
            smooth=True,
            max_n_ticks=5,
            plot_contours=True,
            plot_datapoints=True,
            plot_density=True,
            show_titles=True,
        )

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

        return

    def predict_values(
        self,
        independent: np.array,
        model_parameters: Dict[str, float],
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
            Variance on the model predictions.
        """

        if self.emulator is None:
            raise AttributeError(
                "Please train the emulator with fit_model before attempting "
                "to make predictions."
            )

        if (
            len(self.hyperparameter_samples[:, 0]) < self.samples_for_error
            and self.use_hyperparameter_error
        ):
            raise ValueError(
                "Number of subsamples must be less then the total number of samples"
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

        model, variance = self.emulator.predict(
            y=self.dependent_variables, t=t, return_cov=False, return_var=True
        )

        if self.use_hyperparameter_error:
            # Take a subsample of the MCMC samples
            sample_indices = np.random.choice(
                range(len(self.hyperparameter_samples[:, 0])),
                self.samples_for_error,
                replace=False,
            )
            hyperparameter_error_model_array = np.empty(
                (len(independent), self.samples_for_error), dtype=np.float64
            )
            for index, sample in enumerate(sample_indices):
                self.emulator.set_parameter_vector(
                    self.hyperparameter_samples[index, :]
                )
                hyperparameter_error_model_array[:, sample] = self.emulator.predict(
                    y=self.dependent_variables, t=t, return_cov=False, return_var=False
                )

            # Calculate the variance caused by hyperparameters
            hyper_variance = np.var(hyperparameter_error_model_array, axis=1)
            self.emulator.set_parameter_vector(self.hyperparameter_best_fit)
            variance += hyper_variance

        return model, variance
