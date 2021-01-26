"""
Model parameters container, contains only the parameters
that are part of the differences between models (i.e. not
anything about the individual scaling relations!).
"""

import attr

from typing import Dict, Hashable


@attr.s
class ModelParameters(object):
    """
    Class that contains the parameters of the models, i.e.
    this does not contain any information about individual
    scaling relations. Performs validation on the dictionary
    that is given.

    Parameters
    ----------

    model_parameters, Dict[Hashable, Dict[str, float]]
        Free parameters of the underlying model. This is
        specified as a dictionary with the following
        structure: ``{unique_run_identifier: {parameter_name:
        parameter_value}}``. Here the unique run identifier
        can be anything, but it must be unique between runs.
        An example could be just an integer defining a run
        number. The parameter names must match with
        what is defined in the :class:``ModelSpecification``,
        with the parameter values the specific values taken
        for that individual simulation run. Note that all
        models must have each parameter present, and this
        is checked at the creation time of the ``ModelParameters``
        object.

    Raises
    ------

    AttributeError
        When the parameters do not match between all models.
    """

    model_parameters: Dict[Hashable, Dict[str, float]] = attr.ib()

    @model_parameters.validator
    def _check_model_parameters(self, attribute, value):
        # Basic validation - make sure we have the same
        # parameters in each dictionary.

        # Get an _example set_ of parameters.
        example_parameters = set(next(iter(value.values())).keys())

        for parameter_set in value.values():
            if not example_parameters == set(parameter_set.keys()):
                raise AttributeError(
                    "Models do not all have the same set of parameters."
                )

    def find_closest_model(self, comparison_parameters: Dict[str, float]):
        """
        Finds the closest model currently in this instance of
        ``ModelParameters`` to the set of provided
        ``comparison_parameters``.

        Parameters
        ----------

        comparison_parameters, Dict[str, float]
            Set of comparison parameters. The closest parameters,
            and unique indentifier, of the run within the current
            set of ``model_parameters`` to this point in
            n-dimensional parameter space will be returned.
        
        Returns
        -------

        unique_identifier, Any
            Unique identifier of the closest run.

        closest_parameters, Dict[str, float]
            Model parameters of the closest run.
        """

        raise NotImplementedError

        return
