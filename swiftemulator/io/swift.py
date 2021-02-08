"""
I/O functions for reading in SWIFT simulation data.

Includes functions to read parameters to
instances of :class:``ModelParameters``, and functions
to read model data to instances of :class:``ModelValues``.
"""

from swiftemulator.backend.model_parameters import ModelParameters
from swiftemulator.backend.model_specification import ModelSpecification
from swiftemulator.backend.model_values import ModelValues

from typing import List, Optional, Dict, Hashable, Tuple, Union
from functools import reduce
from pathlib import Path
from math import log10

import yaml
import numpy as np


def load_pipeline_outputs(
    filenames: Dict[Hashable, Path],
    scaling_relations: List[str],
    log_independent: Optional[List[str]] = None,
    log_dependent: Optional[List[str]] = None,
) -> Tuple[Dict[str, ModelValues], Dict[str, Dict[str, Union[str, bool]]]]:
    """
    Loads the pipeline outputs from the provided files, for
    the given specified scaling relations, into a
    :class:``ModelValues`` container.

    Parameters
    ----------

    filenames, Dict[Hashable, Path]
        Paths to files to load data from, with the keys in the
        dictionary the unique identifiers for the models to use
        throughout.

    scaling_relations: List[str]
        Top-level name for the scaling relations (i.e. the top-level
        item in the yaml file, e.g. ``stellar_mass_function_100``).

    log_independent: List[str], optional
        Scaling relations (the same as in ``scaling_relations``) where
        the independent values (given by ``centers`` in the yaml files)
        should be log-scaled (uses ``log10``).

    log_dependent: List[str], optional
        Scaling relations (the same as in ``scaling_relations``) where
        the dependent values (given by ``values`` in the yaml files)
        should be log-scaled (uses ``log10``).

    Returns
    -------

    model_values, Dict[str, ModelValues]
        Dictionary of ``ModelValues`` containers for each scaling relation,
        read from the files. The keys are the names of the scaling relations.

    unit_dict, Dict[str, Dict[str, Union[str, bool]]]
        Dictionary of symbolic units for each scaling relation. Has the
        structure: ``{scaling_relation: {independent: "Msun", dependent: 
        "kpc", log_independent: True, log_dependent: True}}``.
    
    """

    model_values = {scaling_relation: {} for scaling_relation in scaling_relations}

    unit_dict = {scaling_relation: {} for scaling_relation in scaling_relations}

    # Need to search for possible keys within the `lines` dictionary.
    # Priority given by ordering of line_types
    line_types = ["median", "mass_function", "mean"]
    recursive_search = (
        lambda d, k: d.get(k[0], recursive_search(d, k[1:])) if len(k) > 1 else None
    )
    line_search = lambda d: recursive_search(d, line_types)

    for unique_identifier, filename in filenames.items():
        with open(filename, "r") as handle:
            raw_data = yaml.safe_load(handle)

        for scaling_relation in scaling_relations:
            line = line_search(raw_data[scaling_relation]["lines"])

            if line is None:
                continue

            independent = np.array(line["centers"])
            unit_dict[scaling_relation]["independent_units"] = line["centers_units"]

            dependent = np.array(line["values"])
            unit_dict[scaling_relation]["dependent_units"] = line["values_units"]

            dependent_error = np.array(line.get("scatter", np.zeros_like(dependent)))

            if scaling_relation in log_independent:
                independent = np.log10(independent)
                unit_dict[scaling_relation]["log_independent"] = True
            else:
                unit_dict[scaling_relation]["log_independent"] = False

            if scaling_relation in log_dependent:
                # Handle case of dependent errors needing to be logged.
                if dependent_error.ndim > 1:
                    lower = dependent - dependent_error[:0]
                    upper = dependent + dependent_error[:1]
                else:
                    lower = dependent - dependent_error
                    upper = dependent + dependent_error

                dependent = np.log10(dependent)

                upper_diff = np.log10(upper) - dependent
                lower_diff = dependent - np.log10(lower)

                dependent_error = 0.5 * (upper_diff + lower_diff)

                unit_dict[scaling_relation]["log_dependent"] = True
            else:
                unit_dict[scaling_relation]["log_dependent"] = False

                if dependent_error.ndim > 1:
                    dependent_error = np.mean(dependent_error, axis=1)

            model_values[scaling_relation][unique_identifier] = {
                "independent": independent,
                "dependent": dependent,
                "dependent_error": dependent_error,
            }

    return {k: ModelValues(v) for k, v in model_values.items()}, unit_dict


def load_parameter_files(
    filenames: Dict[Hashable, Path],
    parameters: List[str],
    log_parameters: Optional[List[str]] = None,
    parameter_printable_names: Optional[List[str]] = None,
    parameter_limits: Optional[List[List[float]]] = None,
) -> Tuple[ModelSpecification, ModelParameters]:
    """
    Loads information from the parameter files and returns
    the associated model specification and model parameters
    instances.

    Parameters
    ----------

    filenames, Dict[Hashable, Path]
        Paths to the parameter files, keyed by their unique
        identifiers (i.e. those also used in :func:`load_pipeline_outputs`).

    parameters, List[str]
        Parameters to load from the yaml files. Should be specified
        in the same way as the ``--param`` option in SWIFT, i.e.
        in the format ``SectionName:ParameterName``.

    log_parameters, List[str], optional
        Which parameters in the list above should be scaled
        logarithmically.

    parameter_printable_names, List[str], optional
        Optional 'fancy' names for your parameters. These strings will
        be used on any figures generated through swift-emulator. Can
        include LaTeX formatting as in ``matplotlib``.

    parameter_limits, List[List[float]], optional
        The lower and upper limit of the input parameters. Should be
        the same length as ``parameters``, but each item is a list
        of length two, with a lower and upper bound. For example, in
        a two parameter model ``[[0.0, 1.0], [8.3, 9.3]]`` would mean
        that the first parameter would vary between 0.0 and 1.0, with
        the second parameter varying between 8.3 and 9.3. If not provided,
        these will be inferred from the data.
    
    Returns
    -------

    model_specification, ModelSpecification
        Specification for the model based on the parameters that
        have been passed to this function.
    
    model_parameters, ModelParameters
        Model parameter container corresponding to the SWIFT
        parameter files.
    """

    if log_parameters is None:
        log_parameters = []
    else:
        for parameter in log_parameters:
            if not parameter in parameters:
                raise AttributeError(
                    f"Parameter {parameter} requested for logarithmic transform "
                    "not available in main list of parameters."
                )

    model_parameters = {k: None for k in filenames.keys()}

    for unique_identifier, filename in filenames.items():
        with open(filename, "r") as handle:
            full_parameter_file = yaml.load(handle, Loader=yaml.Loader)

        base_parameters = {
            parameter: float(
                reduce(lambda d, k: d.get(k), parameter.split(":"), full_parameter_file)
            )
            for parameter in parameters
        }

        model_parameters[unique_identifier] = {
            parameter: log10(value) if parameter in log_parameters else value
            for parameter, value in base_parameters.items()
        }

    if parameter_limits is None:
        parameter_limits = [
            [
                min([model[parameter] for model in model_parameters.values()]),
                max([model[parameter] for model in model_parameters.values()]),
            ]
            for parameter in parameters
        ]

    model_specification = ModelSpecification(
        number_of_parameters=len(parameters),
        parameter_names=parameters,
        parameter_limits=parameter_limits,
        parameter_printable_names=parameter_printable_names,
    )

    model_parameters = ModelParameters(model_parameters=model_parameters)

    return model_specification, model_parameters
