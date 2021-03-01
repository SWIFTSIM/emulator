"""
Comparison to observational data.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

from velociraptor.observations import ObservationalData

from swiftemulator.backend.model_values import ModelValues

from typing import List, Optional, Dict, Hashable, Union


def continuous_difference(
    independent_A: np.array,
    dependent_A: np.array,
    independent_B: np.array,
    dependent_B: np.array,
    difference_range: Optional[List[float]] = None,
) -> float:
    """
    Calculate the continuous difference between two datasets A and B.
    Effectively calculates a true geometric difference overlap
    between the two (as an L1 norm).

    independent_A: np.array
        Independent variables for dataset A

    dependent_A: np.array
        Dependent variables for dataset A

    independent_B: np.array
        Independent variables for dataset B

    dependent_B: np.array
        Dependent variables for dataset B

    difference_range: List[float], optional
        Range (lower, upper) to calculate the difference overlap
        between. If not present, the whole range of independent_A
        is used.

    Returns
    -------

    difference: float
        The area difference between the two lines.

    Notes
    -----

    This function effectively calculates the difference in a
    linearly interpolated integral between the two lines. At the
    edges of a domain, we simply use the last available value.
    """

    fill_value = lambda x, y: (y[x.argmin()], y[x.argmax()])
    interpolated = lambda x, y: interp1d(
        x=x,
        y=y,
        kind="linear",
        copy=False,
        bounds_error=False,
        fill_value=fill_value(x, y),
    )

    interp_A = interpolated(x=independent_A, y=dependent_A)
    interp_B = interpolated(x=independent_B, y=dependent_B)

    integration_function = lambda x: abs(interp_A(x) - interp_B(x))
    integration_bounds = (
        difference_range
        if difference_range is not None
        else (independent_A.min(), independent_A.max())
    )

    integral, *_ = quad(
        integration_function,
        a=integration_bounds[0],
        b=integration_bounds[1],
    )

    return integral


def continuous_difference_fast(
    independent_A: np.array,
    dependent_A: np.array,
    independent_B: np.array,
    dependent_B: np.array,
    difference_range: Optional[List[float]] = None,
):
    """
    Calculate the continuous difference between two datasets A and B.
    Effectively calculates a true geometric difference overlap
    between the two (as an L1 norm). This is a faster version of the
    'definitely' correct ``continuous_difference``.

    independent_A: np.array
        Independent variables for dataset A

    dependent_A: np.array
        Dependent variables for dataset A

    independent_B: np.array
        Independent variables for dataset B

    dependent_B: np.array
        Dependent variables for dataset B

    difference_range: List[float], optional
        Range (lower, upper) to calculate the difference overlap
        between. If not present, the whole range of independent_A
        is used.

    Returns
    -------

    difference: float
        The area difference between the two lines.

    Notes
    -----

    This function effectively calculates the difference in a
    linearly interpolated integral between the two lines. At the
    edges of a domain, we simply use the last available value.
    """

    difference = 0.0

    sort_A = np.argsort(independent_A)
    sort_B = np.argsort(independent_B)

    raise NotImplementedError

    return difference


def continuous_model_offset_from_observation(
    model_values: ModelValues,
    observation: ObservationalData,
    unit_dict: Dict[str, Union[str, bool]],
    model_difference_range: List[float],
) -> Dict[Hashable, float]:
    """
    Calculate the offset for each model in the model value
    to the observations.

    Parameters
    ----------

    model_values, ModelValues
        ``ModelValues`` container for the scaling relation.

    observation, ObservationalData
        ``ObservationalData`` instance from ``velociraptor-python``
        containing the scaling relation. Must have units that
        correspond to units in the ``unit_dict``, below.

    unit_dict, Dict[str, Dict[str, Union[str, bool]]]
        Dictionary of symbolic units for the scaling relation. Has the
        structure: ``{independent: "Msun", dependent:
        "kpc", log_independent: True, log_dependent: True}``

    model_difference_range: List[float]
        Lower and upper bounds to calculate the model difference over.
        Must be given in the unit system of the model, not of the
        observation.

    Returns
    -------

    model_offsets: Dict[Hashable, float]
        Continuous integral between the data and the observations,
        see :func:`continuous_difference` for more information, keyed
        by the unique identifiers of the models.
    """

    obs_independent = observation.x.to(unit_dict["independent_units"]).value
    obs_dependent = observation.y.to(unit_dict["dependent_units"]).value

    if unit_dict["log_independent"]:
        obs_independent = np.log10(obs_independent)

    if unit_dict["log_dependent"]:
        obs_dependent = np.log10(obs_dependent)

    model_offsets = {}

    for unique_id, relation in model_values.model_values.items():
        model_offsets[unique_id] = continuous_difference(
            independent_A=relation["independent"],
            dependent_A=relation["dependent"],
            independent_B=obs_independent,
            dependent_B=obs_dependent,
            difference_range=model_difference_range,
        )

    return model_offsets
