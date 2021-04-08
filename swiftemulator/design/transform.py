"""
Transformer from an ND array to a model specification object.
"""

from swiftemulator.backend.model_parameters import ModelParameters
from swiftemulator.backend.model_specification import ModelSpecification

import numpy as np

from typing import Optional


def transform_to_model_spec(
    input_array: np.ndarray,
    model_specification: ModelSpecification,
    prefix_unique_id: Optional[str] = None,
) -> ModelParameters:
    """
    Transforms the input nd array (which is of shape n models
    by n parameters) to a model parameters object, by re-scaling the
    parameters according to the specification.

    Parameters
    ----------

    input_array: np.ndarray
        Input array, of shape (number of samples, number of parameters).

    model_specification: ModelSpecification
        Model specification used to rescale the array.

    prefix_unique_id: str, optional
        An optional prefix for the newly generated unique IDs.
        Defaults to no prefix.


    Returns
    -------

    model_parameters: ModelParameters
        A model values container with the re-scaled parameters and
        associated metadata.
    """

    transform = lambda i, l: float((i * (l[1] - l[0])) + l[0])
    prefix = prefix_unique_id if prefix_unique_id is not None else ""
    number_of_samples = len(input_array)

    model_parameters = {
        f"{prefix}{key}": {
            par: transform(input_array[key][i], model_specification.parameter_limits[i])
            for i, par in enumerate(model_specification.parameter_names)
        }
        for key in range(number_of_samples)
    }

    return ModelParameters(model_parameters=model_parameters)
