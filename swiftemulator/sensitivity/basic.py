"""
Basic sensitivity analysis based purely on the model values
at consistent values in the space. No emulation is used to
determine the sensitivity.

A different sensitivity analysis is ran for each dependent
variable, so it is important to ensure that the functions
are evaluated at consistent values.
"""

from swiftemulator.backend.model_parameters import ModelParameters
from swiftemulator.backend.model_values import ModelValues
from swiftemulator.backend.model_specification import ModelSpecification

from typing import Dict, Tuple, Union, Optional

import numpy as np
import matplotlib.pyplot as plt

from SALib.analyze import rbd_fast


def binwise_sensitivity(
    specification: ModelSpecification,
    parameters: ModelParameters,
    values: ModelValues,
) -> Dict[str, np.array]:
    """
    Creates a binwise sensitivity analysis dictionary.

    For each bin in dependent variable a hash is created; these
    are the keys in the returned dictionary.

    Parameters
    ----------

    specification: ModelSpecification
        Model spec; parameter limits must be valid as these feed into
        the sensitivity analysis.

    parameters: ModelParameters
        Parameters; these feed in as the independent variables in the
        sensitivity analysis.

    values: ModelValues
        Dependent variables in the sensitivity analysis.


    Returns
    -------

    sensitivity: Dict[str, np.array]
        Binwise sensitivity analysis, with each array corresponding
        to the parameters in the order as specified by the
        ``specification``. This is the "S1" vector from the
        RBD FAST method.
    """

    # Eventual structure: hash : List[List[parameters]]
    independent = {}
    # Eventual structure: hash : List[values]
    dependent = {}

    hash_function = lambda x: f"{x:1.4g}"

    for id, simulation in values.model_values.items():
        pars = [
            parameters.model_parameters[id][x] for x in specification.parameter_names
        ]
        xs = simulation["independent"]
        ys = simulation["dependent"]

        hashes = map(hash_function, xs)

        for hash, y in zip(hashes, ys):
            try:
                dependent[hash].append(y)
            except KeyError:
                dependent[hash] = [y]

            try:
                independent[hash].append(pars)
            except KeyError:
                independent[hash] = [pars]

    # These dictionaries need to now be transformed into 2D arrays
    # describing the independent and dependent variables for
    # each bin.

    independent = {k: np.array(v) for k, v in independent.items()}
    dependent = {k: np.array(v) for k, v in dependent.items()}

    sensitivities = {
        k: np.array(
            rbd_fast.analyze(
                problem=specification.salib_problem,
                X=independent[k],
                Y=dependent[k],
            )["S1"]
        )
        for k in independent.keys()
    }

    return sensitivities


def plot_binwise_sensitivity(
    specification: ModelSpecification,
    sensitivities: Dict[str, np.array],
    figure: Optional[plt.Figure] = None,
    axes: Optional[plt.Axes] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    cbarlabel: Optional[str] = None,
    cmap: Optional[Union[str, plt.cm.ScalarMappable]] = None,
    vmin: float = -0.25,
    vmax: float = 0.25,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a figure and axis displaying the output of the sensitivity
    analysis.
    """

    if figure is None:
        figure, axes = plt.subplots(
            figsize=(
                8 * (len(sensitivities)) / 20,
                3 * specification.number_of_parameters / 7,
            )
        )

    if cmap is None:
        cmap = "RdBu_r"

    # Convert dictionary to a 2D array ready for colour mapping.
    sensitivity_map = np.zeros((len(sensitivities), specification.number_of_parameters))

    for column, values in enumerate(sensitivities.values()):
        try:
            sensitivity_map[column][:] = values[:]
        except:
            pass

    mappable = axes.imshow(
        sensitivity_map.T, vmin=vmin, vmax=vmax, cmap=cmap, origin="lower"
    )

    axes.set_yticks(range(specification.number_of_parameters))
    axes.set_yticklabels(specification.parameter_printable_names)
    axes.set_xticks(range(len(sensitivities)))
    axes.set_xticklabels(list(sensitivities.keys()), rotation=90)
    figure.colorbar(label=cbarlabel, ax=axes, mappable=mappable)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)

    return figure, axes
