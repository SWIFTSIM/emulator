"""
Generates a latin hypercube ``ModelValues`` container given
the ``ModelSpecification``. Uses :mod:``pyDOE``.
"""

from pyDOE import lhs

from swiftemulator.backend.model_specification import ModelSpecification
from swiftemulator.backend.model_parameters import ModelParameters

from typing import Optional, Dict, Any


def create_hypercube(
    model_specification: ModelSpecification, number_of_samples: int,
) -> ModelParameters:
    """
    Creates a Latin Hypercube model design.

    Parameters
    ----------

    model_specification: ModelSpecification
        Model specification for which to create a latin hypercube
        from.

    number_of_samples: int, optional
        The number of samples to draw; this will be the number
        of input simulations that you wish to create.


    Returns
    -------

    model_parameters: ModelParameters
        A model values container with the prepared latin hypercube.
        Contains methods to visualise the output hypercube.


    Notes
    -----

    Uses :mod:``pyDOE``'s :func:`lhs` function, with the ``correlation``
    method, hence minimising the maximum correlation coefficient.
    """

    samples = lhs(
        n=model_specification.number_of_parameters,
        samples=number_of_samples,
        criterion="corr",
    )

    # Transform the samples to the output space.

    transform = lambda i, l: (i * (l[1] - l[0])) + l[0]

    model_parameters = {
        key: {
            par: transform(samples[key][i], model_specification.parameter_limits[i])
            for i, par in enumerate(model_specification.parameter_names)
        }
        for key in range(number_of_samples)
    }

    return ModelParameters(model_parameters=model_parameters)
