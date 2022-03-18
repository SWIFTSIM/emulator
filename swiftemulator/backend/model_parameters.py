"""
Model parameters container, contains only the parameters
that are part of the differences between models (i.e. not
anything about the individual scaling relations!).
"""

import attr
import numpy as np
import matplotlib.pyplot as plt
import corner
import yaml
import json


from sklearn.neighbors import KDTree
from typing import Dict, Hashable, List, Tuple, Optional, Union, Any
from pathlib import Path

from swiftemulator.backend.model_specification import ModelSpecification


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

    def items(self):
        return self.model_parameters.items()

    def keys(self):
        return self.model_parameters.keys()

    def values(self):
        return self.model_parameters.values()

    def __getitem__(self, key):
        return self.model_parameters[key]

    def __len__(self):
        return len(self.model_parameters)

    def find_closest_model(
        self, comparison_parameters: Dict[str, float], number_of_close_models: int = 1
    ) -> Tuple[List[Hashable], List[Dict[str, float]]]:
        """
        Finds the closest model currently in this instance of
        ``ModelParameters`` to the set of provided
        ``comparison_parameters``, with the option to return
        the closest 'n' sets to the input.

        Parameters
        ----------

        comparison_parameters, Dict[str, float]
            Set of comparison parameters. The closest parameters,
            and unique indentifier, of the run within the current
            set of ``model_parameters`` to this point in
            n-dimensional parameter space will be returned.

        number_of_close_models, int
            Number of closest model that will be returned


        Returns
        -------

        unique_identifier, List[Hashable]
            Unique identifier of the closest run(s).

        closest_parameters, List[Dict[str, float]]
            Model parameters of the closest run(s).
        """

        # Convert the parameter dictionary to a numpy array
        parameter_ordering = list(comparison_parameters.keys())
        parameter_array = []
        model_ordering = []

        for identifier, model in self.model_parameters.items():
            model_ordering.append(identifier)
            parameter_array.append(
                [model[parameter] for parameter in parameter_ordering]
            )

        parameter_array = np.array(parameter_array)

        tree = KDTree(parameter_array, leaf_size=2)
        # Convert the closest point to a numpy array to use for the tree
        search_point = np.array(
            [comparison_parameters[parameter] for parameter in parameter_ordering]
        )

        ind = tree.query(
            search_point.reshape(1, -1),
            k=number_of_close_models,
            return_distance=False,
        )
        # Return in the correct format
        closest_models = [model_ordering[i] for i in ind[0]]
        closest_parameters = [self.model_parameters[i] for i in closest_models]

        return closest_models, closest_parameters

    def plot_model(
        self,
        model_specification: ModelSpecification,
        filename: Optional[Union[Path, str]] = None,
        corner_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Plots the model parameters based on the model specification
        given. Can either be saved to file, or show.

        Parameters
        ----------

        model_specification, ModelSpecification
            Model specification object for this set of parameters.

        filename, Union[str, Path], optional
            Name for the file to which the plot is saved. Optional, if None it
            will show the image.

        corner_kwargs: Dict[str, Any], optional
            Optional key word arguments to pass to `corner` for
            the plotting.

        """

        # Convert internal data to a format that corner can accept

        corner_data = np.empty(
            (len(self.model_parameters), model_specification.number_of_parameters),
            dtype=np.float32,
        )

        for index, model in enumerate(self.model_parameters.values()):
            corner_data[index] = np.array(
                [model[parameter] for parameter in model_specification.parameter_names],
                dtype=np.float32,
            )

        if corner_kwargs is None:
            corner_kwargs = {}

        corner.corner(
            corner_data,
            labels=model_specification.parameter_printable_names,
            range=model_specification.parameter_limits,
            plot_contours=False,
            plot_datapoints=True,
            plot_density=False,
            data_kwargs={"alpha": 1.0},
            **corner_kwargs,
        )

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

    def to_yaml(self, filename: Path):
        """
        Write the model parameters to a YAML file.

        Parameters
        ----------

        filename: Path
            The path to write the file to. This should be a `Path` object,
            but if it is a string it will be automatically converted.

        """

        filename = Path(filename)

        with open(filename, "w") as handle:
            yaml.dump(self.model_parameters, stream=handle)

        return

    @classmethod
    def from_yaml(cls, filename: Path) -> "ModelParameters":
        """
        Generate an instance of :class:`ModelParameters` from a YAML file,
        written to disk using ``to_yaml``.

        Parameters
        ----------

        filename: Path
            The path to read the file from. This should be a ``Path`` object,
            but if it is a string it will be automatically converted.

        Returns
        -------

        model_parameters: ModelParameters
            Instance of ``ModelParameters`` restored from disk.

        """

        filename = Path(filename)

        with open(filename, "r") as handle:
            raw_values = dict(yaml.load(stream=handle, Loader=yaml.FullLoader))

        return cls(model_parameters=raw_values)

    def to_json(self, filename: Path):
        """
        Write the model parameters to a JSON file. Preferred over YAML
        as this is much faster for large datasets.

        Parameters
        ----------

        filename: Path
            The path to write the file to. This should be a `Path` object,
            but if it is a string it will be automatically converted.

        """

        filename = Path(filename)

        with open(filename, "w") as handle:
            json.dump(self.model_parameters, handle)

        return

    @classmethod
    def from_json(cls, filename: Path) -> "ModelParameters":
        """
        Generate an instance of :class:`ModelParameters` from a JSON file,
        written to disk using ``to_json``.

        Parameters
        ----------

        filename: Path
            The path to read the file from. This should be a ``Path`` object,
            but if it is a string it will be automatically converted.

        Returns
        -------

        model_parameters: ModelParameters
            Instance of ``ModelParameters`` restored from disk.

        """

        filename = Path(filename)

        with open(filename, "r") as handle:
            raw_values = dict(json.load(handle))

        return cls(model_parameters=raw_values)
