"""
Visualisation functions for comparison datasets.

Allows you to project a plausibility region for each
parameter cross-correlation.
"""

from swiftemulator.backend.model_parameters import ModelParameters
from swiftemulator.backend.model_specification import ModelSpecification
from swiftemulator.backend.model_values import ModelValues

from typing import Dict, Hashable, Tuple, Iterable, Optional
from matplotlib.colors import Normalize
from swiftsimio.visualisation.projection import scatter
from scipy.stats import binned_statistic_2d

import matplotlib.pyplot as plt
import numpy as np


def visualise_penalties_mean(
    model_specification: ModelSpecification,
    model_parameters: ModelParameters,
    penalties: Dict[Hashable, float],
    norm: Normalize = Normalize(vmin=0.2, vmax=0.7, clip=True),
    remove_ticks: bool = True,
    figsize: Tuple[float] = (7.0, 7.0),
) -> Tuple[plt.Figure, Iterable[plt.Axes]]:
    """
    Visualises the penalties using SPH smoothing for each
    individual model point.

    Parameters
    ----------

    model_specification: ModelSpecification
        The appropriate model specification. Used for the limits
        of the figure.

    model_parameters: ModelParameters
        Parameters of the model, with the appropriate unique IDs.

    penalties: Dict[Hashable, float]
        Penalties for all parameters in ``model_parameters``, with
        the key in this dictionary being the unique IDs.

    norm: Normalize, optional
        A ``matplotlib`` normalisation object. By default this uses
        ``vmin=0.2`` and ``vmax=0.7``.

    remove_ticks: bool, optional
        Remove the axes ticks? This is recommended, as the plot can
        become very cluttered if you don't do this. Default: ``True``.

    figsize: Tuple[float], optional
        The figure size to use. Defaults to 7 inches by 7 inches, the
        size for a ``figure*`` in the MNRAS template.


    Returns
    -------

    fig: Figure
        The figure object.

    axes: np.ndarray[Axes]
        The individual axes.

    Notes
    -----

    You can either change how the figure looks by using the figure
    and axes objects that are returned, or by modifying the
    ``matplotlib`` stylesheet you are currently using.
    """

    number_of_parameters = model_specification.number_of_parameters
    grid_size = number_of_parameters

    fig, axes_grid = plt.subplots(
        grid_size,
        grid_size,
        figsize=figsize,
        squeeze=True,
        sharex="col",
        sharey="row",
    )

    visualisation_size = 2.0 / np.sqrt(len(model_parameters))
    simulation_ordering = list(model_parameters.keys())

    # Build temporary 1D arrays of parameters/offsets in correct ordering
    ordered_penalties = np.array([penalties[x] for x in simulation_ordering])

    limits = model_specification.parameter_limits
    # Parameters must be re-scaled to the range [0,1] for smoothed projection.
    ordered_parameters = [
        (
            np.array(
                [
                    model_parameters.model_parameters[x][parameter]
                    for x in simulation_ordering
                ]
            )
            - limits[index][0]
        )
        / (limits[index][1] - limits[index][0])
        for index, parameter in enumerate(model_specification.parameter_names)
    ]

    smoothing_lengths = np.ones_like(ordered_penalties) * visualisation_size

    for parameter_x, axes_column in enumerate(axes_grid):
        for parameter_y, ax in enumerate(axes_column):
            limits_x = model_specification.parameter_limits[parameter_x]
            limits_y = model_specification.parameter_limits[parameter_y]
            name_x = model_specification.parameter_printable_names[parameter_x]
            name_y = model_specification.parameter_printable_names[parameter_y]

            norm_grid = scatter(
                x=ordered_parameters[parameter_x],
                y=ordered_parameters[parameter_y],
                m=np.ones_like(ordered_penalties),
                h=smoothing_lengths,
                res=512,
            )

            weighted_grid = scatter(
                x=ordered_parameters[parameter_x],
                y=ordered_parameters[parameter_y],
                m=ordered_penalties,
                h=smoothing_lengths,
                res=512,
            )

            norm_grid[norm_grid == 0.0] = 1.0

            ratio_grid = weighted_grid / norm_grid

            im = ax.imshow(
                ratio_grid,
                extent=limits_y + limits_x,
                origin="lower",
                norm=norm,
            )

            ax.set_ylim(*limits_x)
            ax.set_xlim(*limits_y)
            ax.set_ylabel(name_x)
            ax.set_xlabel(name_y)

            if remove_ticks:
                ax.tick_params(
                    axis="both",
                    which="both",
                    bottom=False,
                    left=False,
                    right=False,
                    top=False,
                    labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False,
                )

            # Set square in data reference frame
            ax.set_aspect(1.0 / ax.get_data_ratio())

    for a in axes_grid[:-1, :].flat:
        a.set_xlabel(None)
    for a in axes_grid[:, 1:].flat:
        a.set_ylabel(None)

    return fig, ax


def visualise_penalties_generic_statistic(
    model_specification: ModelSpecification,
    model_parameters: ModelParameters,
    penalties: Dict[Hashable, float],
    statistic: Optional[str] = None,
    norm: Normalize = Normalize(vmin=0.2, vmax=0.7, clip=True),
    remove_ticks: bool = True,
    figsize: Tuple[float] = (7.0, 7.0),
) -> Tuple[plt.Figure, Iterable[plt.Axes]]:
    """
    Visualises the penalties using basic binning.

    Parameters
    ----------

    model_specification: ModelSpecification
        The appropriate model specification. Used for the limits
        of the figure.

    model_parameters: ModelParameters
        Parameters of the model, with the appropriate unique IDs.

    penalties: Dict[Hashable, float]
        Penalties for all parameters in ``model_parameters``, with
        the key in this dictionary being the unique IDs.

    statistic: str, optional
        The statistic that you would like to compute. Allowed values
        are the same as for ``scipy.stats.binned_statistic_2d``.
        Defaults to ``mean``.

    norm: Normalize, optional
        A ``matplotlib`` normalisation object. By default this uses
        ``vmin=0.2`` and ``vmax=0.7``.

    remove_ticks: bool, optional
        Remove the axes ticks? This is recommended, as the plot can
        become very cluttered if you don't do this. Default: ``True``.

    figsize: Tuple[float], optional
        The figure size to use. Defaults to 7 inches by 7 inches, the
        size for a ``figure*`` in the MNRAS template.

    Returns
    -------

    fig: Figure
        The figure object.

    axes: np.ndarray[Axes]
        The individual axes.

    Notes
    -----

    You can either change how the figure looks by using the figure
    and axes objects that are returned, or by modifying the
    ``matplotlib`` stylesheet you are currently using.
    """

    number_of_parameters = model_specification.number_of_parameters
    grid_size = number_of_parameters

    fig, axes_grid = plt.subplots(
        grid_size,
        grid_size,
        figsize=figsize,
        squeeze=True,
        sharex="col",
        sharey="row",
    )

    visualisation_size = 4.0 / np.sqrt(len(model_parameters))
    simulation_ordering = list(model_parameters.keys())

    # Build temporary 1D arrays of parameters/offsets in correct ordering
    ordered_penalties = np.array([penalties[x] for x in simulation_ordering])

    limits = model_specification.parameter_limits
    # Parameters must be re-scaled to the range [0,1] for projection.
    ordered_parameters = [
        (
            np.array(
                [
                    model_parameters.model_parameters[x][parameter]
                    for x in simulation_ordering
                ]
            )
            - limits[index][0]
        )
        / (limits[index][1] - limits[index][0])
        for index, parameter in enumerate(model_specification.parameter_names)
    ]

    bins = np.linspace(0.0, 1.0, int(round(1.0 / visualisation_size)))

    statistic = statistic if statistic is not None else "mean"

    for parameter_x, axes_column in enumerate(axes_grid):
        for parameter_y, ax in enumerate(axes_column):
            limits_x = model_specification.parameter_limits[parameter_x]
            limits_y = model_specification.parameter_limits[parameter_y]
            name_x = model_specification.parameter_printable_names[parameter_x]
            name_y = model_specification.parameter_printable_names[parameter_y]

            grid = binned_statistic_2d(
                x=ordered_parameters[parameter_x],
                y=ordered_parameters[parameter_y],
                values=ordered_penalties,
                statistic=statistic,
                bins=bins,
            )

            im = ax.imshow(
                grid,
                extent=limits_y + limits_x,
                origin="lower",
                norm=norm,
            )

            ax.set_ylim(*limits_x)
            ax.set_xlim(*limits_y)
            ax.set_ylabel(name_x)
            ax.set_xlabel(name_y)

            if remove_ticks:
                ax.tick_params(
                    axis="both",
                    which="both",
                    bottom=False,
                    left=False,
                    right=False,
                    top=False,
                    labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False,
                )

            # Set square in data reference frame
            ax.set_aspect(1.0 / ax.get_data_ratio())

    for a in axes_grid[:-1, :].flat:
        a.set_xlabel(None)
    for a in axes_grid[:, 1:].flat:
        a.set_ylabel(None)

    return fig, ax
