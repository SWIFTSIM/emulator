"""
Object describing the model values, i.e. this uniquely
describes the set of scaling relations present.
"""

import attr
import numpy as np
import yaml
import json

from typing import Dict, Optional, Hashable
from pathlib import Path


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
                if key not in acceptable_keys:
                    raise AttributeError(
                        f"Invalid key {key} in model values container. Choose from "
                        f"one of {acceptable_keys}."
                    )

        return

    def items(self):
        return self.model_values.items()

    def keys(self):
        return self.model_values.keys()

    def values(self):
        return self.model_values.values()

    def __getitem__(self, key):
        return self.model_values[key]

    def __len__(self):
        return len(self.model_values)

    @property
    def number_of_variables(self) -> int:
        """
        Total number of variables present in all models.
        """

        total = sum([len(x["independent"]) for x in self.model_values.values()])

        return total

    def to_yaml(self, filename: Path):
        """
        Write the model values to a YAML file.

        Parameters
        ----------

        filename: Path
            The path to write the file to. This should be a `Path` object,
            but if it is a string it will be automatically converted.

        """

        filename = Path(filename)

        # We must first construct a version of `model_values` that uses lists instead
        # of numpy arrays - yaml doesn't like that!

        listify = lambda x: {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in x.items()
        }

        model_value_no_numpy = {
            uid: listify(model) for uid, model in self.model_values.items()
        }

        with open(filename, "w") as handle:
            yaml.dump(model_value_no_numpy, stream=handle)

        return

    @classmethod
    def from_yaml(cls, filename: Path) -> "ModelValues":
        """
        Generate an instance of :class:`ModelValues` from a YAML file,
        written to disk using `to_yaml`.

        Parameters
        ----------

        filename: Path
            The path to read the file from. This should be a `Path` object,
            but if it is a string it will be automatically converted.

        Returns
        -------

        model_values: ModelValues
            Instance of `ModelValues` restored from disk.

        """

        filename = Path(filename)

        with open(filename, "r") as handle:
            raw_values = dict(yaml.load(stream=handle, Loader=yaml.FullLoader))

        # Now need to re-numpify

        numpyify = lambda x: {
            k: (np.array(v) if isinstance(v, list) else v) for k, v in x.items()
        }

        model_value_numpy = {uid: numpyify(model) for uid, model in raw_values.items()}

        return cls(model_values=model_value_numpy)

    def to_json(self, filename: Path):
        """
        Write the model values to a JSON file. Preferred to YAML as this
        is much faster for large datsets.

        Parameters
        ----------

        filename: Path
            The path to write the file to. This should be a `Path` object,
            but if it is a string it will be automatically converted.

        """

        filename = Path(filename)

        # We must first construct a version of `model_values` that uses lists instead
        # of numpy arrays - yaml doesn't like that!

        listify = lambda x: {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in x.items()
        }

        model_value_no_numpy = {
            uid: listify(model) for uid, model in self.model_values.items()
        }

        with open(filename, "w") as handle:
            json.dump(model_value_no_numpy, handle)

        return

    @classmethod
    def from_json(cls, filename: Path) -> "ModelValues":
        """
        Generate an instance of :class:`ModelValues` from a JSON file,
        written to disk using ``to_json``.

        Parameters
        ----------

        filename: Path
            The path to read the file from. This should be a `Path` object,
            but if it is a string it will be automatically converted.

        Returns
        -------

        model_values: ModelValues
            Instance of ``ModelValues`` restored from disk.

        """

        filename = Path(filename)

        with open(filename, "r") as handle:
            raw_values = dict(json.load(handle))

        # Now need to re-numpify

        numpyify = lambda x: {
            k: (np.array(v) if isinstance(v, list) else v) for k, v in x.items()
        }

        model_value_numpy = {uid: numpyify(model) for uid, model in raw_values.items()}

        return cls(model_values=model_value_numpy)
