"""
Sub-module that uses the created emulators to re-sample the parameter
space completely, effectively creating a higher 'resolution'
(in sub-grid parameters) hypercube.
"""

import numpy as np

from swiftemulator.backend.model_specification import ModelSpecification
from swiftemulator.backend.model_values import ModelValues
from swiftemulator.backend.model_parameters import ModelParameters
from swiftemulator.design.random import create_cube

from typing import Dict, Any, Tuple, Optional


def mock_hypercube(
    emulator,
    model_specification: ModelSpecification,
    samples: int,
    predict_values_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[ModelValues, ModelParameters]:
    """
    Create a mocked version of the cube, interpolated at random
    points using the emulator.

    Parameters
    ----------

    emulator
        An emulator object that provides a `predict_values` function.

    model_spec: ModelSpecification
        A model specification for your chosen model. The cube
        will be generated for points within the ranges specified here.

    samples: int
        Number of samples to create within your model specification.

    predict_value_kwargs: dict, optional
        Keyword arguments to pass to ``predict_values`` on the emulator
        object.

    Returns
    -------

    values: ModelValues
        Model values container with the predictions from the provided emulator
        within a new cube.

    parameters: ModelParameters
        New model parameters generated in a random cube, corresponding to
        the unique identifiers in ``values``.

    Notes
    -----

    The unique identifiers for the new simulations are prefixed with
    ``emulated_`` to prevent confusion when comparing with 'real' data.
    Samples are generated at all of the independent variable points
    present within the provided emulator's data.
    """

    predict_values_kwargs = (
        {} if predict_values_kwargs is None else predict_values_kwargs
    )

    parameters = create_cube(
        model_specification=model_specification,
        number_of_samples=samples,
        prefix_unique_id="emulated_",
    )

    independent_variables = np.unique(
        [
            item
            for uid in emulator.model_values.model_values.keys()
            for item in emulator.model_values.model_values[uid]["independent"]
        ]
    )

    emulated_models = {}

    for uid, pars in parameters.model_parameters.items():
        dep, dep_err = emulator.predict_values(
            independent=independent_variables,
            model_parameters=pars,
            **predict_values_kwargs,
        )

        emulated_models[uid] = {
            "independent": independent_variables,
            "dependent": dep,
            "dependent_error": dep_err,
        }

    return ModelValues(model_values=emulated_models), parameters


def mock_sweep(
    emulator,
    model_specification: ModelSpecification,
    samples: int,
    sweep_parameter: str,
    center_point: Dict[str, float],
    predict_values_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[ModelValues, ModelParameters]:
    """
    Create a mocked version of the cube, interpolated evenly spaced
    along one dimension.

    Parameters
    ----------

    emulator
        An emulator object that provides a `predict_values` function.

    model_spec: ModelSpecification
        A model specification for your chosen model. The cube
        will be generated for points within the ranges specified here.

    samples: int
        Number of samples to create within your model specification.

    sweep_parameter: str
        Parameter to sweep along, from minimum to maximum in the
        ``model_spec``.

    center_point: str
        Model parameters for the center point of the sweep. The other
        model parameters will remain as these values throughout the sweep.

    predict_value_kwargs: dict, optional
        Keyword arguments to pass to ``predict_values`` on the emulator
        object.

    Returns
    -------

    values: ModelValues
        Model values container with the predictions from the provided emulator
        within a new cube.

    parameters: ModelParameters
        New model parameters generated in a sweep, corresponding to
        the unique identifiers in ``values``.

    Notes
    -----

    The unique identifiers for the new simulations are prefixed with
    ``emulated_`` to prevent confusion when comparing with 'real' data.
    Samples are generated at all of the independent variable points
    present within the provided emulator's data.
    """

    predict_values_kwargs = (
        {} if predict_values_kwargs is None else predict_values_kwargs
    )

    sweep_parameter_index = model_specification.parameter_names.index(sweep_parameter)

    sweep_values = np.linspace(
        *model_specification.parameter_limits[sweep_parameter_index], samples
    )

    parameters = ModelParameters(
        {
            f"emulated_{key}": {
                k: v if k != sweep_parameter else sweep_values[key]
                for k, v in center_point.items()
            }
            for key in range(samples)
        }
    )

    independent_variables = np.unique(
        [
            item
            for uid in emulator.model_values.model_values.keys()
            for item in emulator.model_values.model_values[uid]["independent"]
        ]
    )

    emulated_models = {}

    for uid, pars in parameters.model_parameters.items():
        dep, dep_err = emulator.predict_values(
            independent=independent_variables,
            model_parameters=pars,
            **predict_values_kwargs,
        )

        emulated_models[uid] = {
            "independent": independent_variables,
            "dependent": dep,
            "dependent_error": dep_err,
        }

    return ModelValues(model_values=emulated_models), parameters
