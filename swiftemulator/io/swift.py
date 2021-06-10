"""
I/O functions for reading in SWIFT simulation data.

Includes functions to read parameters to
instances of :class:``ModelParameters``, and functions
to read model data to instances of :class:``ModelValues``.

Also includes functions to write out :class:``ModelParameters``
as files, based on a base parameter file.
"""

from swiftemulator.backend.model_parameters import ModelParameters
from swiftemulator.backend.model_specification import ModelSpecification
from swiftemulator.backend.model_values import ModelValues

from swiftemulator.io.error import convert_dependent_error_to_standard_error_using_edges

from typing import List, Optional, Dict, Hashable, Tuple, Union, Callable
from functools import reduce
from pathlib import Path
from math import log10

import yaml
import numpy as np


def _validate_bins(
    bin_centers: np.array,
    bin_edges: np.array,
) -> np.array:
    """
    Validates the pipeline output bins. Note that the pipeline will
    claim to output all viable bins, not taking into account the
    fact that some will not be written out as actual centers if there
    is no data in those bins.

    Parameters
    ----------

    bin_centers: np.array
        Centers of the associated bin edges.

    bin_edges: np.array
        A list of _possible_ bin edges, with some not being used
        as acutal bins in ``bin_centers``

    Returns
    -------

    validated_bin_edges: np.array
        Bin edges that are associated with the input bin_edges.

    Notes
    -----

    The current implementation assumes that missing bins contain
    no items, and that the left hand bin is always valid
    """

    try:
        valid_bins = np.digitize(bin_centers, bin_edges, right=True)

        return bin_edges[np.append(valid_bins, 0)]
    except ValueError:
        # Some broken outputs have 2 dimensional bin_xs; these
        # are mass functions that should not be corrected anyway.
        # This is linked to the issue in the velociraptor-python
        # repository at:
        # https://github.com/SWIFTSIM/velociraptor-python/issues/65
        return np.append(bin_centers, bin_centers[-1] * 2.0)


def load_pipeline_outputs(
    filenames: Dict[Hashable, Path],
    scaling_relations: List[str],
    log_independent: Optional[List[str]] = None,
    log_dependent: Optional[List[str]] = None,
    use_mean_sampling_errors: Optional[Dict[str, str]] = None,
    use_median_sampling_errors: Optional[Dict[str, str]] = None,
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

    use_mean_sampling_errors: Dict[str, str], optional
        Use sampling errors for the keys in the dictionaries, using the
        histograms specified as the values. This is useful for correcting
        mean lines that have their scatter given as percentile ranges
        which may include physical scatter in the scaling relation.

    use_median_sampling_errors: Dict[str, str], optional
        Use sampling errors for the keys in the dictionaries, using the
        histograms specified as the values. This is useful for correcting
        median lines that have their scatter given as percentile ranges
        which may include physical scatter in the scaling relation.
        This acts the same as the mean sampling errors, but includes the
        correction factor of 1.253.

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

    use_mean_sampling_errors = (
        {} if use_mean_sampling_errors is None else use_mean_sampling_errors
    )
    use_median_sampling_errors = (
        {} if use_median_sampling_errors is None else use_median_sampling_errors
    )

    histograms_to_read = set(
        list(use_mean_sampling_errors.values())
        + list(use_median_sampling_errors.values())
    )
    histograms = {histogram: {} for histogram in histograms_to_read}

    unit_dict = {scaling_relation: {} for scaling_relation in scaling_relations}

    # Need to search for possible keys within the `lines` dictionary.
    # Priority given by ordering of line_types
    line_types = [
        "median",
        "mass_function",
        "mean",
        "adaptive_mass_function",
        "histogram",
    ]

    recursive_search = (
        lambda d, k: d.get(k[0], recursive_search(d, k[1:])) if len(k) > 0 else None
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

            bin_edges = _validate_bins(
                bin_centers=independent,
                bin_edges=np.array(line["bins_x"]),
            )

            dependent_error = np.array(line.get("scatter", np.zeros_like(dependent)))

            if scaling_relation in log_independent:
                independent = np.log10(independent)
                bin_edges = np.log10(bin_edges)
                unit_dict[scaling_relation]["log_independent"] = True
            else:
                unit_dict[scaling_relation]["log_independent"] = False

            if scaling_relation in log_dependent:
                # Handle case of dependent errors needing to be logged.
                if dependent_error.ndim > 1:
                    lower = dependent - dependent_error[0, :]
                    upper = dependent + dependent_error[1, :]
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
                    dependent_error = np.mean(dependent_error, axis=0)

            model_values[scaling_relation][unique_identifier] = {
                "independent": independent,
                "dependent": dependent,
                "dependent_error": dependent_error,
                "bin_edges": bin_edges,
            }

        for histogram in histograms_to_read:
            line = line_search(raw_data[histogram]["lines"])

            independent = np.array(line["centers"])
            dependent = np.array(line["values"])

            bin_edges = _validate_bins(
                bin_centers=independent,
                bin_edges=np.array(line["bins_x"]),
            )

            histograms[histogram][unique_identifier] = {
                "independent": independent,
                "dependent": dependent,
                "bin_edges": bin_edges,
            }

    # Now correct all mean and median lines to use their associated sampling
    # errors, rather than the scatter that is provided.

    for scaling_relation, histogram in use_mean_sampling_errors.items():
        for unique_id in filenames.keys():
            model_hist = histograms[histogram][unique_id]
            model_scaling_relation = model_values[scaling_relation][unique_id]

            if scaling_relation in log_independent:
                histogram_bin_edges = np.log10(model_hist["bin_edges"])
            else:
                histogram_bin_edges = model_hist["bin_edges"]

            standard_error = convert_dependent_error_to_standard_error_using_edges(
                scaling_relation=model_scaling_relation["dependent"],
                scaling_relation_bin_edges=model_scaling_relation["bin_edges"],
                histogram=model_hist["dependent"],
                histogram_bin_edges=histogram_bin_edges,
                log_independent=scaling_relation in log_independent,
                log_dependent=scaling_relation in log_dependent,
                correction_factor=1.0,
            )

            # Un-2dify the standard error (for now!) as GPE cannot support
            # 2D errors.
            if scaling_relation in log_dependent:
                standard_error = 0.5 * (standard_error[0, :] + standard_error[1, :])

            model_values[scaling_relation][unique_id][
                "dependent_error"
            ] = standard_error

    # Median exactly the same as the code above but using 1.253

    for scaling_relation, histogram in use_median_sampling_errors.items():
        for unique_id in filenames.keys():
            model_hist = histograms[histogram][unique_id]
            model_scaling_relation = model_values[scaling_relation][unique_id]

            if scaling_relation in log_independent:
                histogram_bin_edges = np.log10(model_hist["bin_edges"])
            else:
                histogram_bin_edges = model_hist["bin_edges"]

            standard_error = convert_dependent_error_to_standard_error_using_edges(
                scaling_relation=model_scaling_relation["dependent"],
                scaling_relation_bin_edges=model_scaling_relation["bin_edges"],
                histogram=model_hist["dependent"],
                histogram_bin_edges=histogram_bin_edges,
                log_independent=scaling_relation in log_independent,
                log_dependent=scaling_relation in log_dependent,
                correction_factor=1.253,
            )

            # Un-2dify the standard error (for now!) as GPE cannot support
            # 2D errors.
            if scaling_relation in log_dependent:
                standard_error = 0.5 * (standard_error[0, :] + standard_error[1, :])

            model_values[scaling_relation][unique_id][
                "dependent_error"
            ] = standard_error

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

    read_parameters = {k: None for k in filenames.keys()}

    for unique_identifier, filename in filenames.items():
        with open(filename, "r") as handle:
            full_parameter_file = yaml.load(handle, Loader=yaml.Loader)

        base_parameters = {
            parameter: float(
                reduce(lambda d, k: d.get(k), parameter.split(":"), full_parameter_file)
            )
            for parameter in parameters
        }

        read_parameters[unique_identifier] = {
            parameter: log10(value) if parameter in log_parameters else value
            for parameter, value in base_parameters.items()
        }

    if parameter_limits is None:
        parameter_limits = [
            [
                min([model[parameter] for model in read_parameters.values()]),
                max([model[parameter] for model in read_parameters.values()]),
            ]
            for parameter in parameters
        ]

    model_specification = ModelSpecification(
        number_of_parameters=len(parameters),
        parameter_names=parameters,
        parameter_limits=parameter_limits,
        parameter_printable_names=parameter_printable_names,
    )

    model_parameters = ModelParameters(model_parameters=read_parameters)

    return model_specification, model_parameters


def write_parameter_files(
    filenames: Dict[Hashable, Path],
    model_parameters: ModelParameters,
    parameter_transforms: Optional[Dict[str, Callable]] = None,
    base_parameter_file: Optional[Path] = None,
):
    """
    Writes parameter files, containing the parameters from a
    :class:`ModelParameters` instance, based on a base parameter
    file.

    Parameters
    ----------

    filenames: Dict[Hashable, Path]
        Dictionary stating where to write each parameter file, based
        upon the unique identifiers of each run in ``model_parameters``.

    model_parameters: ModelParameters
        Varied parameters to write in the output files.

    parameter_transforms: Dict[str, Callable], optional
        Parameter transformation functions for transforming parameter values
        before writing. Parameters may be generated (and emulated) in a
        space that is very different to their meaning in the code. Hence,
        this parameter allows for a transformation (for instance, a logrithmic
        transformation). For each parameter that should be transformed (keys)
        there should be a function taking the emulated value, transforming it
        into the code value. For instance, if a parameter is emulated in
        logarithmic space, this should be ``lambda x: 10**x``.

    base_parameter_file: Path, optional
        Base parameter file to read. The parameters specified in
        ``model_parameters`` will be overwritten when writing each
        individual file, but the rest will remain the same.

    Notes
    -----

    Also changes the value of ``MetaData:run_name`` to the unique
    identifiers.
    """

    if base_parameter_file is not None:
        with open(base_parameter_file, "r") as handle:
            base_parameters = yaml.load(handle, Loader=yaml.Loader)
    else:
        base_parameters = {}

    if parameter_transforms is None:
        parameter_transforms = {}

    for identifier, filename in filenames.items():
        parameters = base_parameters.copy()

        for parameter, value in {
            "MetaData:run_name": str(identifier),
            **model_parameters.model_parameters[identifier],
        }.items():
            section, key = parameter.split(":")

            transform = parameter_transforms.get(parameter, lambda x: x)

            try:
                parameters[section][key] = transform(value)
            except KeyError:
                parameters[section] = {}
                parameters[section][key] = transform(value)

        with open(filename, "w") as handle:
            yaml.dump(parameters, handle, default_flow_style=False)

    return
