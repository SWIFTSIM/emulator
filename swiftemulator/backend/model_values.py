"""
Object describing the model values, i.e. this uniquely
describes the set of scaling relations present.
"""

import attr
import numpy as np

from typing import Dict, Optional, Hashable


@attr.s
class ModelValues(object):
    """
    Set of model values for a given scaling relation.

    Parameters
    ----------

    model_values, Dict[Hashable, Dict[str, Optional[np.array]]]
        Model values for this given scaling relation. Must have the
        following structure: ``{unique_identifier: {"independent":
        np.array, "dependent": np.array, "dependent_errors": Optional[
        np.array]}}``, with the ``unique_identifiers`` the same as
        those that were specified within the :class:`ModelParameters`
        object. The dependent errors are optional, but if present
        must be of the same length as the dependent variables. Both
        independent and dependent are required for every model value
        that is present. If a unique identifier is present in the
        model parameters, but not in the values, it is simply left
        out of the emulation step.

    Raises
    ------

    AttributeError
        If the model values do not conform to the required
        specification.
    """

    model_values: Dict[Hashable, Dict[str, Optional[np.array]]] = attr.ib()

    @model_values.validator
    def _check_model_values(self, attribute, value):
        """
        Checks the model values to ensure that all
        required fields are present.
        """

        for unique_id, scaling_relation in value.items():
            if not (
                "independent" in scaling_relation.keys()
                or "dependent" in scaling_relation.keys()
            ):
                raise AttributeError(
                    "independent and dependent must be available for all models, "
                    f"missing for model {unique_id}."
                )

            if "dependent_error" in scaling_relation.keys():
                if not len(scaling_relation["dependent_error"]) == len(
                    scaling_relation["dependent"]
                ):
                    raise AttributeError(
                        "dependent_error present but not the same length as "
                        f"dependent for model {unique_id}."
                    )

            acceptable_keys = ["independent", "dependent", "dependent_error"]

            for key in scaling_relation.keys():
                if key not in :
                    raise AttributeError(
                        f"Invalid key {key} in model values container. Choose from "
                        f"one of {acceptable_keys}."
                    )

        return

    @property
    def number_of_variables(self) -> int:
        """
        Total number of variables present in all models.
        """

        total = sum([len(x["independent"]) for x in self.model_values.values()])

        return total
