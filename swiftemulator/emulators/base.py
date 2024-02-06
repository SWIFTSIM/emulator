"""
Base for all emulators. This :class:`BaseEmulator` class is a specification
for how all emulators should be implemented, but does not implement a
specific model itself (all methods raise ``NotImplementedError`` by default).
Controls on the behaviour of the emulator are passed as initialiser
arguments, and the data is passed as arguments to fit_model.
"""

import attr
import copy
import numpy as np

from typing import Hashable, List, Optional, Dict

from swiftemulator.backend.model_parameters import ModelParameters
from swiftemulator.backend.model_specification import ModelSpecification
from swiftemulator.backend.model_values import ModelValues


@attr.s
class BaseEmulator(object):
    """
    Base emulator for training models. Initialisation parameters
    are used in ``fit_model`` to specify additional parameters
    to the model.
    """

    ordering: Optional[List[Hashable]] = None
    parameter_order: Optional[List[str]] = None

    model_specification: Optional[ModelSpecification] = None
    model_parameters: Optional[ModelParameters] = None
    model_values: Optional[ModelValues] = None

    independent_variables: Optional[np.array] = None
    dependent_variables: Optional[np.array] = None
    dependent_variable_errors: Optional[np.array] = None

    emulator = None

    def _build_arrays(
        self,
        model_specification: ModelSpecification,
        model_parameters: ModelParameters,
        model_values: ModelValues,
    ):
        """
        Builds the internal arrays based on the given model specification,
        parameters, and values.
        """

        self.model_specification = model_specification
        self.model_parameters = model_parameters
        self.model_values = model_values

        self.independent_variables = None
        self.dependent_variables = None
        self.dependent_variable_errors = None

        raise NotImplementedError

    def fit_model(
        self,
        model_specification: ModelSpecification,
        model_parameters: ModelParameters,
        model_values: ModelValues,
    ):
        """
        Fits a model to the independent and dependent variables given by
        the model spec, parameters, and values.
        """

        self._build_arrays()

        self.emulator = None

        raise NotImplementedError

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
            (disregarding the additional 'independent' model parameters, below.)

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

        if self.emulator is None:
            raise AttributeError(
                "Please train the emulator with fit_model before attempting "
                "to make predictions."
            )

        raise NotImplementedError

    def interactive_plot(self, x: np.array, xlabel: str = "", ylabel: str = ""):
        """
        Generates an interactive plot over which shows the emulator predictions
        for the input data passed to this method
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        fig, ax = plt.subplots()
        model_specification = self.model_specification
        param_means = {}
        sliders = []
        n_param = model_specification.number_of_parameters
        fig.subplots_adjust(bottom=0.12 + n_param * 0.1)
        for i in range(n_param):
            # Extracting information needed for slider
            name = model_specification.parameter_names[i]
            lo_lim = sorted(model_specification.parameter_limits[i])[0]
            hi_lim = sorted(model_specification.parameter_limits[i])[1]
            param_means[name] = (lo_lim + hi_lim) / 2

            # Adding slider
            if model_specification.parameter_printable_names:
                name = model_specification.parameter_printable_names[i]
            slider_ax = fig.add_axes([0.35, i * 0.1, 0.3, 0.1])
            slider = Slider(
                ax=slider_ax,
                label=name,
                valmin=lo_lim,
                valmax=hi_lim,
                valinit=(lo_lim + hi_lim) / 2,
            )
            sliders.append(slider)

        # Setting up initial value
        pred, pred_var = self.predict_values(x, param_means)
        (line,) = ax.plot(x, pred)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Define and enable update function
        def update(val):
            params = {
                model_specification.parameter_names[i]: sliders[i].val
                for i in range(n_param)
            }
            pred, pred_var = self.predict_values(x, params)
            line.set_ydata(pred)

        for slider in sliders:
            slider.on_changed(update)

        plt.show()
        plt.close()
