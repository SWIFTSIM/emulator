"""
Penalty functions and their specifications.
"""

import attr
import unyt
import numpy as np
import matplotlib.pyplot as plt

from velociraptor.observations import ObservationalData
from scipy.interpolate import interp1d

from swiftemulator.backend.model_values import ModelValues

from typing import Dict, Hashable, Optional, Union, Callable, List


@attr.s
class PenaltyCalculator(object):
    """
    Base class for the penalty functions.

    Extend this with your own, following the following pattern:

    1. Configuration parameters for the penalty function
       are taken as initialisation parameters to the class.
    2. The observational data set to be matched to is passed
       to register_observation.
    3. The penalty function is calculated using the penalty()
       method, taking the exact arguments that are taken here,
       for an individual model.

    Provided for convenience is penalties() which calculates the
    penalty function for all data in a `ModelValues` container.
    """

    observation: ObservationalData
    interpolator_values: interp1d
    interpolator_error: Optional[interp1d]

    log_independent: bool
    log_dependent: bool
    independent_units: unyt.unyt_quantity
    dependent_units: unyt.unyt_quantity

    def register_observation(
        self,
        observation: ObservationalData,
        log_independent: bool,
        log_dependent: bool,
        independent_units: unyt.unyt_quantity,
        dependent_units: unyt.unyt_quantity,
    ) -> None:
        """
        Registers the observation for use in `penalty` with the class.

        Parameters
        ----------

        observation: ObservationalData
            Instance of the velociraptor observational data used
            for comparisons.

        log_independent: bool
            Take the base-10 log of the independent data before comparison?

        log_dependent: bool
            Take the base-10 log of the dependent data before comparison?

        independent_units: unyt.unyt_quantity
            The units that the model was calculated in (independent)

        dependent_units: unyt.unyt_quantity
            The units that the model was calculated in (dependent)

        """

        self.observation = observation
        self.log_independent = log_independent
        self.log_dependent = log_dependent
        self.independent_units = independent_units
        self.dependent_units = dependent_units

        self.observation_interpolation()

        return

    def observation_interpolation(self):
        """
        Produces the interpolation for the internal observation.
        """

        x = self.observation.x.to(self.independent_units)
        y = self.observation.y.to(self.dependent_units)

        if self.log_independent:
            x = np.log10(x.value)

        if self.log_dependent:
            y = np.log10(y.value)

        fill_value = lambda x, y: (y[x.argmin()], y[x.argmax()])

        self.interpolator_values = interp1d(
            x=x,
            y=y,
            kind="linear",
            copy=False,
            bounds_error=False,
            fill_value=fill_value(x, y),
        )

        # Set observational error to None for now.
        self.interpolator_error = None

        return

    def penalty(
        self,
        independent: np.array,
        dependent: np.array,
        dependent_error: Optional[np.array] = None,
    ) -> List[float]:
        """
        Calculate the penalty function, relative to the observational
        data, for this model. It is highly recommended that you evaluate
        the model at the same independent variables as the observational
        data. The observational data is linearly interpolated to find a
        prediction at the independent variables that you provide.

        independent: np.array
            The independent data.

        dependent: np.array
            The dependent data for comparison.

        dependent_error: np.array, optional
            The dependent errors, for comparison.

        Returns
        -------

        penalty, List[float]
            The penalties for this model, between 0 and 1 each.
        """

        raise NotImplementedError

    def penalties(
        self, model: ModelValues, collate_with: Callable
    ) -> Dict[Hashable, float]:
        """
        Calculate the penalty function for all models in the
        model values container.

        It is highly recommended that you evaluate the model at the same
        independent variables as the observational data. The observational
        data is linearly interpolated to find a prediction at the independent
        variables that you provide.

        Parameters
        ----------

        model: ModelValues
            The set of model (values) to calculate the penalty
            function for.

        collate_with: Callable
            A function that takes a numpy array and returns the
            'global' penalty for a model given the input for all
            of the valid points in the array. Examples could be
            ``np.max``, ``np.mean``, ``np.median``.


        Returns
        -------

        penalties: Dict[Hashable, float]
            Penalty functions for each of the models, with the key
            being the unique identifier.
        """

        penalties = {}

        for unique_id, data in model.model_values.items():
            independent = data["independent"]
            dependent = data["dependent"]
            dependent_error = data.get("dependent_error", None)

            penalties[unique_id] = collate_with(
                self.penalty(
                    independent=independent,
                    dependent=dependent,
                    dependent_error=dependent_error,
                )
            )

        return penalties

    def plot_penalty(
        self,
        low_independent: Union[unyt.unyt_quantity, float],
        high_independent: Union[unyt.unyt_quantity, float],
        low_dependent: Union[unyt.unyt_quantity, float],
        high_dependent: Union[unyt.unyt_quantity, float],
        filename: str,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        resolution: int = 128,
        marker: Optional[str] = None,
    ):
        """
        Create a figure of the penalty function, over the limits
        given. Limits are given in log space if logged, linear
        space if not (with units required in that case).
        """

        fig, ax = plt.subplots()

        if not self.log_independent:
            low_independent = low_independent.to(self.independent_units)
            high_independent = high_independent.to(self.independent_units)

        if not self.log_dependent:
            low_dependent = low_dependent.to(self.dependent_units)
            high_dependent = high_dependent.to(self.dependent_units)

        x = np.linspace(low_independent, high_independent, resolution)
        y = np.linspace(low_dependent, high_dependent, resolution)

        xs, ys = np.meshgrid(x, y)
        output = np.empty_like(xs)

        for index, (independent, dependent) in enumerate(zip(xs.flat, ys.flat)):
            output.flat[index] = self.penalty(
                independent=np.array([independent]),
                dependent=np.array([dependent]),
                dependent_error=None,
            )

        mappable = ax.pcolormesh(xs, ys, output, vmin=0.0, vmax=1.0, rasterized=True)
        fig.colorbar(mappable=mappable, ax=ax, label="Penalty Function")

        # Plot obs data
        x = self.observation.x.to(self.independent_units)
        y = self.observation.y.to(self.dependent_units)

        if self.log_independent:
            x = np.log10(x.value)

        if self.log_dependent:
            y = np.log10(y.value)

        ax.plot(x, y, linestyle="dashed", marker=marker)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        ax.set_xlim(low_independent, high_independent)
        ax.set_ylim(low_dependent, high_dependent)

        fig.savefig(filename)

        return


@attr.s
class L1PenaltyCalculator(PenaltyCalculator):
    """
    Penalty calculator for an L1-type norm, i.e. a linear
    penalty function away from the data. This penalty function is
    capped after a (vertical) distance, provided with units
    if the provided observation is used in linear space, or
    provided as a logarithmic offset in dex if in log space.

    Parameters
    ----------

    offset: Union[unyt.unyt_quantity, float]
        The vertical offset at which to set the L1 norm to the
        maximum. This is required as the penalty function is
        not allowed to be unlimited.

    lower: Union[unyt.unyt_quantity, float]
        The lowest independent value to calculate the model
        offset at.

    upper: Union[unyt.unyt_quantity, float]
        The highest independent value to calculate the model
        offset at.
    """

    offset: Union[unyt.unyt_quantity, float] = attr.ib()
    lower: Union[unyt.unyt_quantity, float] = attr.ib()
    upper: Union[unyt.unyt_quantity, float] = attr.ib()

    def observation_interpolation(self):
        super().observation_interpolation()

        # Convert limits to sensible units and log them
        # if necessary
        if not self.log_independent:
            self.lower.convert_to_units(self.independent_units)
            self.upper.convert_to_units(self.independent_units)

        if not self.log_dependent:
            self.offset.convert_to_units(self.dependent_units)

        return

    def penalty(
        self,
        independent: np.array,
        dependent: np.array,
        dependent_error: Optional[np.array] = None,
    ) -> float:
        valid_data_mask = np.logical_and(
            independent >= self.lower, independent < self.upper
        )

        number_of_valid_points = valid_data_mask.sum()

        if number_of_valid_points == 0:
            return 0.0

        obs_dependent = self.interpolator_values(independent[valid_data_mask])

        penalties = np.abs(obs_dependent - dependent[valid_data_mask]) / self.offset

        return np.minimum(penalties, 1.0)


@attr.s
class L1PenaltyCalculatorOneSided(PenaltyCalculator):
    """
    Penalty calculator for an L1-type norm, i.e. a linear
    penalty function away from the data, but one-sided only.
    Values above/below the line are given a maximal
    penalty.

    This penalty function is capped after a (vertical) distance, provided
    with units if the provided observation is used in linear space, or
    provided as a logarithmic offset in dex if in log space.

    Parameters
    ----------

    offset: Union[unyt.unyt_quantity, float]
        The vertical offset at which to set the L1 norm to the
        maximum. This is required as the penalty function is
        not allowed to be unlimited.

    maximum_penalty: str
        Give the maximum penalty above or below the line. Accepted
        values are "above" or "below" as strings.

    lower: Union[unyt.unyt_quantity, float]
        The lowest independent value to calculate the model
        offset at.

    upper: Union[unyt.unyt_quantity, float]
        The highest independent value to calculate the model
        offset at.
    """

    offset: Union[unyt.unyt_quantity, float] = attr.ib()
    maximum_penalty: str = attr.ib()
    lower: Union[unyt.unyt_quantity, float] = attr.ib()
    upper: Union[unyt.unyt_quantity, float] = attr.ib()

    @maximum_penalty.validator
    def _check_maximum_penalty(self, attribute, value):
        valid = ["above", "below"]
        if value not in valid:
            raise AttributeError("maximum_penalty must be one of above or below.")

    def observation_interpolation(self):
        super().observation_interpolation()

        # Convert limits to sensible units and log them
        # if necessary
        if not self.log_independent:
            self.lower.convert_to_units(self.independent_units)
            self.upper.convert_to_units(self.independent_units)

        if not self.log_dependent:
            self.offset.convert_to_units(self.dependent_units)

        return

    def penalty(
        self,
        independent: np.array,
        dependent: np.array,
        dependent_error: Optional[np.array] = None,
    ) -> float:
        valid_data_mask = np.logical_and(
            independent >= self.lower, independent < self.upper
        )

        number_of_valid_points = valid_data_mask.sum()

        if number_of_valid_points == 0:
            return 0.0

        obs_dependent = self.interpolator_values(independent[valid_data_mask])

        penalties = np.abs(obs_dependent - dependent[valid_data_mask]) / self.offset

        if self.maximum_penalty == "above":
            penalties[dependent[valid_data_mask] > obs_dependent] = 1.0
        elif self.maximum_penalty == "below":
            penalties[dependent[valid_data_mask] < obs_dependent] = 1.0

        return np.minimum(penalties, 1.0)


@attr.s
class L2PenaltyCalculator(PenaltyCalculator):
    """
    Penalty calculator for an L2-type norm, i.e. a square
    penalty function away from the data. This penalty function is
    capped after a (vertical) distance, provided with units
    if the provided observation is used in linear space, or
    provided as a logarithmic offset in dex if in log space.

    Parameters
    ----------

    offset: Union[unyt.unyt_quantity, float]
        The vertical offset at which to set the L1 norm to the
        maximum. This is required as the penalty function is
        not allowed to be unlimited.

    lower: Union[unyt.unyt_quantity, float]
        The lowest independent value to calculate the model
        offset at.

    upper: Union[unyt.unyt_quantity, float]
        The highest independent value to calculate the model
        offset at.
    """

    offset: Union[unyt.unyt_quantity, float] = attr.ib()
    lower: Union[unyt.unyt_quantity, float] = attr.ib()
    upper: Union[unyt.unyt_quantity, float] = attr.ib()

    def observation_interpolation(self):
        super().observation_interpolation()

        # Convert limits to sensible units and log them
        # if necessary
        if not self.log_independent:
            self.lower.convert_to_units(self.independent_units)
            self.upper.convert_to_units(self.independent_units)

        if not self.log_dependent:
            self.offset.convert_to_units(self.dependent_units)

        return

    def penalty(
        self,
        independent: np.array,
        dependent: np.array,
        dependent_error: Optional[np.array] = None,
    ) -> float:
        valid_data_mask = np.logical_and(
            independent >= self.lower, independent < self.upper
        )

        number_of_valid_points = valid_data_mask.sum()

        if number_of_valid_points == 0:
            return 0.0

        obs_dependent = self.interpolator_values(independent[valid_data_mask])

        penalties = (obs_dependent - dependent[valid_data_mask]) ** 2 / self.offset ** 2

        return np.minimum(penalties, 1.0)


@attr.s
class L1VariablePenaltyCalculator(PenaltyCalculator):
    """
    Penalty calculator for an L1-type norm, i.e. a linear
    penalty function away from the data. This penalty function is
    capped after a (vertical) distance, provided with units
    if the provided observation is used in linear space, or
    provided as a logarithmic offset in dex if in log space.

    In this version the offset is variable, using a logistic
    curve.

    Parameters
    ----------

    offset_lower: Union[unyt.unyt_quantity, float]
        The vertical offset at which to set the L1 norm to the
        maximum, at the lower end of the independent range.

    offset_upper: Union[unyt.unyt_quantity, float]
        The vertical offset at which to set the L1 norm to the
        maximum, at the upper end of the independent range.

    offset_transition: Union[unyt.unyt_quantity, float]
        The independent variable at which you would like the
        offset to transition from ``offset_lower`` to
        ``offset_upper``.

    transition_width: Union[unyt.unyt_quantity, float]
        The width of the transition between offsets, centered
        around ``offset_transition``.

    lower: Union[unyt.unyt_quantity, float]
        The lowest independent value to calculate the model
        offset at.

    upper: Union[unyt.unyt_quantity, float]
        The highest independent value to calculate the model
        offset at.

    offset_below_above_ratio: float, optional
        Ratio of the allowed offset below or above the data. If this
        takes a value of less than 1.0, models below the data are penalised
        more (by that factor) than models above. Default: 1.0
    """

    offset_lower: Union[unyt.unyt_quantity, float] = attr.ib()
    offset_upper: Union[unyt.unyt_quantity, float] = attr.ib()
    offset_transition: Union[unyt.unyt_quantity, float] = attr.ib()
    transition_width: Union[unyt.unyt_quantity, float] = attr.ib()
    lower: Union[unyt.unyt_quantity, float] = attr.ib()
    upper: Union[unyt.unyt_quantity, float] = attr.ib()
    offset_below_above_ratio: float = attr.ib(default=1.0)

    def observation_interpolation(self):
        super().observation_interpolation()

        # Convert limits to sensible units and log them
        # if necessary
        if not self.log_independent:
            self.lower.convert_to_units(self.independent_units)
            self.upper.convert_to_units(self.independent_units)
            self.offset_transition.convert_to_units(self.independent_units)
            self.transition_width.convert_to_units(self.independent_units)

        if not self.log_dependent:
            self.offset_lower.convert_to_units(self.dependent_units)
            self.offset_upper.convert_to_units(self.dependent_units)

        return

    def penalty(
        self,
        independent: np.array,
        dependent: np.array,
        dependent_error: Optional[np.array] = None,
    ) -> float:
        valid_data_mask = np.logical_and(
            independent >= self.lower, independent < self.upper
        )

        number_of_valid_points = valid_data_mask.sum()

        if number_of_valid_points == 0:
            return 0.0

        offsets = self.offset_lower + (self.offset_upper - self.offset_lower) * (
            1.0
            / (
                1.0
                + np.exp(
                    (self.offset_transition - independent[valid_data_mask])
                    / self.transition_width
                )
            )
        )

        obs_dependent = self.interpolator_values(independent[valid_data_mask])
        valid_and_low_mask = dependent[valid_data_mask] < obs_dependent

        offsets[valid_and_low_mask] *= self.offset_below_above_ratio

        penalties = np.abs(obs_dependent - dependent[valid_data_mask]) / offsets

        return np.minimum(penalties, 1.0)


@attr.s
class L1SqueezePenaltyCalculator(PenaltyCalculator):
    """
    Penalty calculator for an L1-type norm, i.e. a linear
    penalty function away from the data. This penalty function is
    capped after a (vertical) distance, provided with units
    if the provided observation is used in linear space, or
    provided as a logarithmic offset in dex if in log space.

    In this version the offset is variable, with it being
    'squeezed' at a point over a width.

    Parameters
    ----------

    offset_squeeze: Union[unyt.unyt_quantity, float]
        The vertical offset at which to set the L1 norm to the
        maximum at the pinch point.

    offset_normal: Union[unyt.unyt_quantity, float]
        The usual vertical offset at which to set the L1 norm
        to the maximum.

    offset_transition: Union[unyt.unyt_quantity, float]
        The independent variable at which you would like the
        offset to transition from ``offset_lower`` to
        ``offset_upper``.

    transition_width: Union[unyt.unyt_quantity, float]
        The width of the transition between offsets, centered
        around ``offset_transition``.

    lower: Union[unyt.unyt_quantity, float]
        The lowest independent value to calculate the model
        offset at.

    upper: Union[unyt.unyt_quantity, float]
        The highest independent value to calculate the model
        offset at.

    offset_below_above_ratio: float, optional
        Ratio of the allowed offset below or above the data. If this
        takes a value of less than 1.0, models below the data are penalised
        more (by that factor) than models above. Default: 1.0
    """

    offset_squeeze: Union[unyt.unyt_quantity, float] = attr.ib()
    offset_normal: Union[unyt.unyt_quantity, float] = attr.ib()
    offset_transition: Union[unyt.unyt_quantity, float] = attr.ib()
    transition_width: Union[unyt.unyt_quantity, float] = attr.ib()
    lower: Union[unyt.unyt_quantity, float] = attr.ib()
    upper: Union[unyt.unyt_quantity, float] = attr.ib()
    offset_below_above_ratio: float = attr.ib(default=1.0)

    def observation_interpolation(self):
        super().observation_interpolation()

        # Convert limits to sensible units and log them
        # if necessary
        if not self.log_independent:
            self.lower.convert_to_units(self.independent_units)
            self.upper.convert_to_units(self.independent_units)
            self.offset_transition.convert_to_units(self.independent_units)
            self.transition_width.convert_to_units(self.independent_units)

        if not self.log_dependent:
            self.offset_squeeze.convert_to_units(self.dependent_units)
            self.offset_normal.convert_to_units(self.dependent_units)

        return

    def penalty(
        self,
        independent: np.array,
        dependent: np.array,
        dependent_error: Optional[np.array] = None,
    ) -> float:
        valid_data_mask = np.logical_and(
            independent >= self.lower, independent < self.upper
        )

        number_of_valid_points = valid_data_mask.sum()

        if number_of_valid_points == 0:
            return 0.0

        offsets = self.offset_normal - (self.offset_normal - self.offset_squeeze) * (
            np.exp(
                -0.5
                * (
                    (
                        (independent[valid_data_mask] - self.offset_transition)
                        / self.transition_width
                    )
                    ** 2
                )
            )
        )

        obs_dependent = self.interpolator_values(independent[valid_data_mask])
        valid_and_low_mask = dependent[valid_data_mask] < obs_dependent

        offsets[valid_and_low_mask] *= self.offset_below_above_ratio

        penalties = np.abs(obs_dependent - dependent[valid_data_mask]) / offsets

        return np.minimum(penalties, 1.0)


@attr.s
class GaussianDataErrorsPenaltyCalculator(PenaltyCalculator):
    """
    Penalty calculator for observations that include errors.
    This penalty function uses a Gaussian distribution around
    the data, based on the observational errors. Capped at a
    input number of sigmas away from the data.

    Parameters
    ----------

    sigma_max: Union[unyt.unyt_quantity, float]
        The number of sigmas at which the function is capped.

    lower: Union[unyt.unyt_quantity, float]
        The lowest independent value to calculate the model
        offset at.

    upper: Union[unyt.unyt_quantity, float]
        The highest independent value to calculate the model
        offset at.
    """

    sigma_max: Union[unyt.unyt_quantity, float] = attr.ib()
    lower: Union[unyt.unyt_quantity, float] = attr.ib()
    upper: Union[unyt.unyt_quantity, float] = attr.ib()

    error_interpolator_values: interp1d

    def observation_interpolation(self):
        super().observation_interpolation()

        x = self.observation.x.to(self.independent_units)
        y_scatter = self.observation.y_scatter.to(self.dependent_units)

        if self.log_independent:
            x = np.log10(x.value)

        # Propagate errors to log space if needed
        if self.log_independent:
            y = self.observation.y.to(self.dependent_units)
            y_scatter = np.abs(y_scatter / (y * np.log(10)))

        # Account for unsymmetric errors
        if y_scatter.ndim > 1:
            y_scatter = np.mean(y_scatter, axis=0)

        fill_value = lambda x, y_scatter: (y_scatter[x.argmin()], y_scatter[x.argmax()])

        self.error_interpolator_values = interp1d(
            x=x,
            y=y_scatter,
            kind="linear",
            copy=False,
            bounds_error=False,
            fill_value=fill_value(x, y_scatter),
        )

        # Convert limits to sensible units and log them
        # if necessary
        if not self.log_independent:
            self.lower.convert_to_units(self.independent_units)
            self.upper.convert_to_units(self.independent_units)

        return

    def penalty(
        self,
        independent: np.array,
        dependent: np.array,
        dependent_error: Optional[np.array] = None,
    ) -> List[float]:

        valid_data_mask = np.logical_and(
            independent >= self.lower, independent < self.upper
        )

        number_of_valid_points = valid_data_mask.sum()

        if number_of_valid_points == 0:
            return 0.0

        obs_dependent = self.interpolator_values(independent[valid_data_mask])
        obs_dependent_errors = self.error_interpolator_values(
            independent[valid_data_mask]
        )

        penalties = 1.0 - np.exp(
            -0.5
            * (
                (dependent[valid_data_mask] - obs_dependent) ** 2
                / (obs_dependent_errors ** 2)
            )
        )

        sigma_max_mask_value = 1.0 - np.exp(-0.5 * (self.sigma_max ** 2))
        penalties[penalties > sigma_max_mask_value] = 1.0

        return np.minimum(penalties, 1.0)


@attr.s
class GaussianPercentErrorsPenaltyCalculator(PenaltyCalculator):
    """
    Penalty calculator that that uses Gaussian errors with
    widths based on the percentages difference between the model
    and the data.

    Parameters
    ----------

    percent_error: float
        percent error that sets the one-sigma deviation,
        in units of percent (0-100).

    sigma_max: float
        The number of sigmas at which the function is capped.

    lower: Union[unyt.unyt_quantity, float]
        The lowest independent value to calculate the model
        offset at.

    upper: Union[unyt.unyt_quantity, float]
        The highest independent value to calculate the model
        offset at.
    """

    percent_error: float = attr.ib()
    sigma_max: float = attr.ib()
    lower: Union[unyt.unyt_quantity, float] = attr.ib()
    upper: Union[unyt.unyt_quantity, float] = attr.ib()

    def observation_interpolation(self):
        super().observation_interpolation()

        # Convert limits to sensible units and log them
        # if necessary
        if not self.log_independent:
            self.lower.convert_to_units(self.independent_units)
            self.upper.convert_to_units(self.independent_units)

        return

    def penalty(
        self,
        independent: np.array,
        dependent: np.array,
        dependent_error: Optional[np.array] = None,
    ) -> List[float]:

        valid_data_mask = np.logical_and(
            independent >= self.lower, independent < self.upper
        )

        number_of_valid_points = valid_data_mask.sum()

        if number_of_valid_points == 0:
            return 0.0

        obs_dependent = self.interpolator_values(independent[valid_data_mask])

        penalties = 1.0 - np.exp(
            -0.5
            * (
                (dependent[valid_data_mask] - obs_dependent) ** 2
                / ((obs_dependent * (self.percent_error / 100)) ** 2)
            )
        )

        sigma_max_mask_value = 1.0 - np.exp(-0.5 * (self.sigma_max ** 2))
        penalties[penalties > sigma_max_mask_value] = 1.0

        return np.minimum(penalties, 1.0)


@attr.s
class GaussianDataErrorsPercentFloorPenaltyCalculator(PenaltyCalculator):
    """
    Penalty calculator that that uses Gaussian errors based
    on the observational data. Includes a floor based on a
    percent error. It will pick the worst out of the two.
    This is meant as a way to not fit better then the
    emulator allows, while also not constraining stronger
    than observations.

    Parameters
    ----------

    percent_error: float
        percent error that sets the one-sigma deviation,
        in units of percent (0-100).

    sigma_max: float
        The number of sigmas at which the function is capped.
    lower: Union[unyt.unyt_quantity, float]
        The lowest independent value to calculate the model
        offset at.

    upper: Union[unyt.unyt_quantity, float]
        The highest independent value to calculate the model
        offset at.
    """

    percent_error: float = attr.ib()
    sigma_max: float = attr.ib()
    lower: Union[unyt.unyt_quantity, float] = attr.ib()
    upper: Union[unyt.unyt_quantity, float] = attr.ib()

    error_interpolator_values: interp1d

    def observation_interpolation(self):
        super().observation_interpolation()

        x = self.observation.x.to(self.independent_units)
        y_scatter = self.observation.y_scatter.to(self.dependent_units)

        if self.log_independent:
            x = np.log10(x.value)

        # Propagate errors to log space if needed
        if self.log_independent:
            y = self.observation.y.to(self.dependent_units)
            y_scatter = np.abs(y_scatter / (y * np.log(10)))

        # Account for unsymmetric errors
        if y_scatter.ndim > 1:
            y_scatter = np.mean(y_scatter, axis=0)

        fill_value = lambda x, y_scatter: (y_scatter[x.argmin()], y_scatter[x.argmax()])

        self.error_interpolator_values = interp1d(
            x=x,
            y=y_scatter,
            kind="linear",
            copy=False,
            bounds_error=False,
            fill_value=fill_value(x, y_scatter),
        )

        # Convert limits to sensible units and log them
        # if necessary
        if not self.log_independent:
            self.lower.convert_to_units(self.independent_units)
            self.upper.convert_to_units(self.independent_units)

        return

    def penalty(
        self,
        independent: np.array,
        dependent: np.array,
        dependent_error: Optional[np.array] = None,
    ) -> List[float]:

        valid_data_mask = np.logical_and(
            independent >= self.lower, independent < self.upper
        )

        number_of_valid_points = valid_data_mask.sum()

        if number_of_valid_points == 0:
            return 0.0

        obs_dependent = self.interpolator_values(independent[valid_data_mask])
        obs_dependent_errors = self.error_interpolator_values(
            independent[valid_data_mask]
        )

        penalties = 1.0 - np.exp(
            -0.5
            * (
                (dependent[valid_data_mask] - obs_dependent) ** 2
                / (
                    obs_dependent_errors ** 2
                    + (obs_dependent * (self.percent_error / 100)) ** 2
                )
            )
        )

        sigma_max_mask_value = 1.0 - np.exp(-0.5 * (self.sigma_max ** 2))
        penalties[penalties > sigma_max_mask_value] = 1.0

        return np.minimum(penalties, 1.0)


@attr.s
class GaussianWeightedDataErrorsPercentFloorPenaltyCalculator(PenaltyCalculator):
    """
    Penalty calculator that that uses Gaussian errors based
    on the observational data. Includes a floor based on a
    percent error. It will pick the worst out of the two.
    This is meant as a way to not fit better then the
    emulator allows, while also not constraining stronger
    than observations.

    Parameters
    ----------

    percent_error: float
        percent error that sets the one-sigma deviation,
        in units of percent (0-100).

    sigma_max: float
        The number of sigmas at which the function is capped.

    lower: Union[unyt.unyt_quantity, float]
        The lowest independent value to calculate the model
        offset at.

    upper: Union[unyt.unyt_quantity, float]
        The highest independent value to calculate the model
        offset at.

    weight: A general weight that scales the entire range of
        errors, but keeps relative weights intact.
    """

    percent_error: float = attr.ib()
    sigma_max: float = attr.ib()
    weight: float = attr.ib()
    lower: Union[unyt.unyt_quantity, float] = attr.ib()
    upper: Union[unyt.unyt_quantity, float] = attr.ib()

    error_interpolator_values: interp1d

    def observation_interpolation(self):
        super().observation_interpolation()

        x = self.observation.x.to(self.independent_units)
        y_scatter = self.observation.y_scatter.to(self.dependent_units)

        if self.log_independent:
            x = np.log10(x.value)

        # Propagate errors to log space if needed
        if self.log_independent:
            y = self.observation.y.to(self.dependent_units)
            y_scatter = np.abs(y_scatter / (y * np.log(10)))

        # Account for unsymmetric errors
        if y_scatter.ndim > 1:
            y_scatter = np.mean(y_scatter, axis=0)

        fill_value = lambda x, y_scatter: (y_scatter[x.argmin()], y_scatter[x.argmax()])

        self.error_interpolator_values = interp1d(
            x=x,
            y=y_scatter,
            kind="linear",
            copy=False,
            bounds_error=False,
            fill_value=fill_value(x, y_scatter),
        )

        # Convert limits to sensible units and log them
        # if necessary
        if not self.log_independent:
            self.lower.convert_to_units(self.independent_units)
            self.upper.convert_to_units(self.independent_units)

        return

    def penalty(
        self,
        independent: np.array,
        dependent: np.array,
        dependent_error: Optional[np.array] = None,
    ) -> List[float]:

        valid_data_mask = np.logical_and(
            independent >= self.lower, independent < self.upper
        )

        number_of_valid_points = valid_data_mask.sum()

        if number_of_valid_points == 0:
            return 0.0

        obs_dependent = self.interpolator_values(independent[valid_data_mask])
        obs_dependent_errors = self.error_interpolator_values(
            independent[valid_data_mask]
        )

        penalties = 1.0 - np.exp(
            -0.5
            * (
                (dependent[valid_data_mask] - obs_dependent) ** 2
                / (
                    obs_dependent_errors ** 2
                    + (obs_dependent * (self.percent_error / 100)) ** 2
                )
            )
        )

        sigma_max_mask_value = 1.0 - np.exp(-0.5 * (self.sigma_max ** 2))
        penalties[penalties > sigma_max_mask_value] = 1.0

        return np.minimum(penalties, 1.0)
