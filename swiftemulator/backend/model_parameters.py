"""
Model parameters container, contains only the parameters
that are part of the differences between models (i.e. not
anything about the individual scaling relations!).
"""

import attr

from typing import Dict, Any


@attr.s
class ModelParmeters(object):
    """
    Class that contains the parameters of the models, i.e.
    this does not contain any information about individual
    scaling relations. Performs validation on the dictionary
    that is given.
    """

    model_parameters: Dict[Any, Dict[str, float]] = attr.ib()

    @model_parameters.validator
    def _check_model_parameters(self, attribute, value):
        # Basic validation - make sure we have the same
        # parameters in each dictionary.

        # Get an _example set_ of parameters.
        example_parameters = set(next(iter(value.values())).keys())
        comparison_parameters = set()

        for parameter_set in value.values():
            comparison_parameters.update(parameter_set.keys())

        if example_parameters != comparison_parameters:
            raise AttributeError("Models do not all have the same set of parameters.")

