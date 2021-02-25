"""
Generates a completely random design, by using the numpy random command.
"""

from swiftemulator.backend.model_specification import ModelSpecification
from swiftemulator.backend.model_parameters import ModelParameters
from swiftemulator.design.transform import transform_to_model_spec

from typing import Optional, Dict, Any, Optional

import numpy as np


def create_cube(
    model_specification: ModelSpecification,
    number_of_samples: int,
    prefix_unique_id: Optional[str] = None,
) -> ModelParameters:
    """
    Creates a random hypercube model design.

    Parameters
    ----------

    model_specification: ModelSpecification
        Model specification for which to create a latin hypercube
        from.

    number_of_samples: int
        The number of samples to draw; this will be the number
        of input simulations that you wish to create.

    prefix_unique_id: str, optional
        An optional prefix for the newly generated unique IDs.
        Defaults to no prefix.


    Returns
    -------

    model_parameters: ModelParameters
        A model values container with the prepared latin hypercube.
        Contains methods to visualise the output hypercube.


    Notes
    -----

    Uses numpy's random methods to generate a completely random (i.e.
    no guarantee of a nice even distribution) hypercube
    """

    samples = np.random.rand(
        number_of_samples, model_specification.number_of_parameters
    )

    return transform_to_model_spec(
        input_array=samples,
        model_specification=model_specification,
        prefix_unique_id=prefix_unique_id,
    )
