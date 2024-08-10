"""
A gaussian process emulator that uses _multiple_ internal emulators to
better predict functions that contain a 'break'.
"""

import attr
import numpy as np
import george
import math

from typing import Hashable, List, Optional, Dict

from swiftemulator.emulators.base import BaseEmulator
from swiftemulator.emulators.gaussian_process import GaussianProcessEmulator
from swiftemulator.backend.model_parameters import ModelParameters
from swiftemulator.backend.model_specification import ModelSpecification
from swiftemulator.backend.model_values import ModelValues
from swiftemulator.mean_models import MeanModel

from scipy.optimize import minimize


@attr.s
class MultipleGaussianProcessEmulator(BaseEmulator):
    """
    Generator for emulators for individual scaling relations,
    using multiple trained gaussian processes regression
    instances and linear models under the hood.

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

    independent_regions, List[List[float]]
        The regions over which to construct independent emulators.
        None can be used in the first and last element to specify
        there are no boundaries to overlap. Must be monotonically
        increasing. Overlaps between regions are allowed and
        predicted values will be a weighted linear combination
        of both. For example, you could use
        ``[[None, 1.0], [0.7, 2.0], [1.7, None]]`` for data that
        ran from 0.0 to 3.0 in the independent variable.
        Regions should not overlap more than once. This isn't checked,
        but will break the code.
    """

    kernel: Optional[george.kernels.Kernel] = attr.ib(default=None)
    mean_model: Optional[MeanModel] = attr.ib(default=None)
    independent_regions: Optional[List[List[float]]] = attr.ib(default=[[None, None]])

    model_specification: Optional[ModelSpecification] = None
    model_parameters: Optional[ModelParameters] = None
    model_values: Optional[ModelValues] = None
    model_values_regions: Optional[List[ModelValues]] = None

    emulators: Optional[List[george.GP]] = None

    def _build_arrays(
        self,
        model_specification: ModelSpecification,
        model_parameters: ModelParameters,
        model_values: ModelValues,
    ):
        """
        Builds the arrays for passing to sub-emulators.

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

        # Here we must take our full instances and convert
        # them into separate instances only containing the sub
        # independent regions
        unique_identifiers = list(model_values.keys())

        values = [{} for _ in self.independent_regions]

        for unique_id in unique_identifiers:
            raw_values = model_values[unique_id]

            ind = raw_values["independent"]
            dep = raw_values["dependent"]

            no_error = False

            try:
                err = raw_values["dependent_error"]
            except KeyError:
                no_error = True

            for index, (low, high) in enumerate(self.independent_regions):
                mask = np.logical_and(
                    ind > low if low is not None else np.ones_like(ind).astype(bool),
                    ind < high if high is not None else np.ones_like(ind).astype(bool),
                )

                values[index][unique_id] = {
                    "independent": ind[mask],
                    "dependent": dep[mask],
                }

                if not no_error:
                    values[index][unique_id]["dependent_error"] = err[mask]

        self.model_values_regions = [ModelValues(x) for x in values]

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

        if self.independent_variables is None:
            # Creates independent_variables, dependent_variables.
            self._build_arrays(
                model_specification=model_specification,
                model_parameters=model_parameters,
                model_values=model_values,
            )

        self.emulators = [
            GaussianProcessEmulator(kernel=self.kernel, mean_model=self.mean_model)
            for _ in self.independent_regions
        ]

        for gp, model_vals in zip(self.emulators, self.model_values_regions):
            gp.fit_model(
                model_specification=model_specification,
                model_parameters=model_parameters,
                model_values=model_vals,
            )

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
            Errors on the model predictions.

        Notes
        -----

        This will use the originally defined regions and overlaps will
        be calculated by using the weighted linear sum corresponding
        to the independent variable's distance to the adjacent boundary.
        The errors use a weighted square sum.
        """

        if self.emulators is None:
            raise AttributeError(
                "Please train the emulator with fit_model before attempting "
                "to make predictions."
            )

        # First, do individual predictions.

        inputs = []
        output = []
        output_error = []

        for index, (low, high) in enumerate(self.independent_regions):
            mask = np.logical_and(
                independent > low
                if low is not None
                else np.ones_like(independent).astype(bool),
                independent < high
                if high is not None
                else np.ones_like(independent).astype(bool),
            )

            predicted, errors = self.emulators[index].predict_values(
                independent=independent[mask], model_parameters=model_parameters
            )

            inputs.append(list(independent[mask]))
            output.append(list(predicted))
            output_error.append(list(errors))

        # Now that we've predicted it all, we need to explicitly deal
        # with overlap and non-overlap.

        overlap_ranges = {}

        for index in range(1, len(self.independent_regions)):
            left = self.independent_regions[index][0]
            right = self.independent_regions[index - 1][1]

            if right is None or left is None:
                continue
            elif right > left:
                overlap_ranges[index - 1] = [left, right]

        dependent_predictions = np.empty_like(independent)
        dependent_prediction_errors = np.empty_like(independent)

        current_emulator = 0

        for index, x in enumerate(independent):
            if x not in inputs[current_emulator]:
                current_emulator += 1

            # Is it in the prior overlap?
            low, high = overlap_ranges.get(current_emulator - 1, [float("inf")] * 2)

            if low <= x <= high:
                # We have already counted this independent variable.
                continue

            # Is it in this emulator's overlap?
            low, high = overlap_ranges.get(current_emulator, [float("inf")] * 2)

            if low <= x <= high:
                dependent_index_left = inputs[current_emulator].index(x)
                dependent_index_right = inputs[current_emulator + 1].index(x)

                ind_left = inputs[current_emulator][dependent_index_left]
                ind_right = inputs[current_emulator + 1][dependent_index_right]

                left_weight = (high - x) / (high - low)
                right_weight = (x - low) / (high - low)

                dependent_left = output[current_emulator][dependent_index_left]
                dependent_right = output[current_emulator + 1][dependent_index_right]

                dependent_error_left = output_error[current_emulator][
                    dependent_index_left
                ]
                dependent_error_right = output_error[current_emulator][
                    dependent_index_right
                ]

                dependent_predictions[index] = (
                    dependent_left * left_weight + dependent_right * right_weight
                )
                dependent_prediction_errors[index] = math.sqrt(
                    left_weight * dependent_error_left * dependent_error_left
                    + right_weight * dependent_error_right * dependent_error_right
                )
            else:
                # Easy!
                dependent_index = inputs[current_emulator].index(x)
                dependent_predictions[index] = output[current_emulator][dependent_index]
                dependent_prediction_errors[index] = output_error[current_emulator][
                    dependent_index
                ]

        return dependent_predictions, dependent_prediction_errors
