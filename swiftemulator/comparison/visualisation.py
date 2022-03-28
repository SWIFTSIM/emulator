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
from scipy.stats import binned_statistic_2d

try:
    from swiftsimio.visualisation.projection import scatter

    swiftsimio_available = True
except (ImportError, ModuleNotFoundError):
    swiftsimio_available = False

import matplotlib.pyplot as plt
import numpy as np


def visualise_penalties_mean(
    model_specification: ModelSpecification,
    model_parameters: ModelParameters,
    penalties: Dict[Hashable, float],
    norm: Normalize = Normalize(vmin=0.2, vmax=0.7, clip=True),
    remove_ticks: bool = True,
    figsize: Optional[Tuple[float]] = None,
    use_parameters: Optional[Iterable[str]] = None,
    use_colorbar: Optional[bool] = False,
    highlight_model: Optional[Hashable] = None,
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

    use_parameters: Iterable[str], optional
        The parameters to include in the figure. If not provided, all
        parameters in the ``model_specification`` are used.

    use_colorbar: Bool, optional
        Include a colorbar? Default: False

    highlight_model: Hashable, optional
        The model unique ID to highlight. If not provided, no model is
        highlighted.

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

    if use_parameters is None:
        use_parameters = model_specification.parameter_names

    if figsize is None:
        if use_colorbar:
            figsize = (7.0, 7.4)
        else:
            figsize = (7.0, 7.0)

    parameter_indices = [
        model_specification.parameter_names.index(x) for x in use_parameters
    ]

    number_of_parameters = len(use_parameters)
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

    if highlight_model is not None:
        highlight_index = simulation_ordering.index(highlight_model)

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
        for index, parameter in zip(parameter_indices, use_parameters)
    ]

    smoothing_lengths = np.ones_like(ordered_penalties) * visualisation_size

    for parameter_x, axes_column in zip(parameter_indices, axes_grid):
        for parameter_y, ax in zip(parameter_indices, axes_column):
            limits_x = model_specification.parameter_limits[parameter_x]
            limits_y = model_specification.parameter_limits[parameter_y]
            name_x = model_specification.parameter_printable_names[parameter_x]
            name_y = model_specification.parameter_printable_names[parameter_y]

            is_center_line = parameter_x == parameter_y
            do_not_plot = is_center_line and remove_ticks

            if not do_not_plot:
                if swiftsimio_available:
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
                else:
                    norm_grid, _, _, _ = binned_statistic_2d(
                        x=ordered_parameters[parameter_x],
                        y=ordered_parameters[parameter_y],
                        values=ordered_penalties,
                        statistic="mean",
                        bins=16,
                    )

                im = ax.imshow(
                    ratio_grid.T,
                    extent=limits_x + limits_y,
                    origin="lower",
                    norm=norm,
                    rasterized=True,
                )

                if highlight_model is not None:
                    highlight_x = ordered_parameters[parameter_x][highlight_index]
                    highlight_y = ordered_parameters[parameter_y][highlight_index]

                    # Need to re-scale from 0->1 to 'real' space

                    highlight_x *= limits_x[1] - limits_x[0]
                    highlight_x += limits_x[0]

                    highlight_y *= limits_y[1] - limits_y[0]
                    highlight_y += limits_y[0]

                    ax.scatter(
                        highlight_x,
                        highlight_y,
                        color="white",
                        edgecolor="black",
                    )

                ax.set_xlim(*limits_x)
                ax.set_ylim(*limits_y)

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
                if is_center_line:
                    ax.text(
                        0.5,
                        0.5,
                        f"{limits_x[0]:3.3f} <\n{name_x}\n< {limits_x[1]:3.3f}",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                    )

            else:
                ax.set_xlabel(name_x)
                ax.set_ylabel(name_y)

            # Set square in data reference frame
            ax.set_aspect(1.0 / ax.get_data_ratio())

    if use_colorbar:
        fig.colorbar(
            im,
            ax=axes_grid.ravel().tolist(),
            orientation="horizontal",
            label="Mean penalty along line of sight",
        )

    for a in axes_grid[:-1, :].flat:
        a.set_xlabel(None)
    for a in axes_grid[:, 1:].flat:
        a.set_ylabel(None)

    # As of matplotlib 3.3.4, with a large number of sub-plots this hangs...
    if grid_size > 4:
        fig.constrained_layout = False
        fig.subplots_adjust(0, 0, 1, 1, 0.005, 0.005)

    return fig, ax


def visualise_penalties_generic_statistic(
    model_specification: ModelSpecification,
    model_parameters: ModelParameters,
    penalties: Dict[Hashable, float],
    statistic: Optional[str] = None,
    norm: Normalize = Normalize(vmin=0.2, vmax=0.7, clip=True),
    remove_ticks: bool = True,
    figsize: Optional[Tuple[float]] = None,
    use_parameters: Optional[Iterable[str]] = None,
    use_colorbar: Optional[bool] = False,
    highlight_model: Optional[Hashable] = None,
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

    use_parameters: Iterable[str], optional
        The parameters to include in the figure. If not provided, all
        parameters in the ``model_specification`` are used.

    use_colorbar: Bool, optional
        Include a colorbar? Default: False.

    highlight_model: Hashable, optional
        The model unique ID to highlight. If not provided, no model is
        highlighted.


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

    if use_parameters is None:
        use_parameters = model_specification.parameter_names

    if figsize is None:
        if use_colorbar:
            figsize = (7.0, 7.4)
        else:
            figsize = (7.0, 7.0)

    parameter_indices = [
        model_specification.parameter_names.index(x) for x in use_parameters
    ]

    number_of_parameters = len(use_parameters)
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

    if highlight_model is not None:
        highlight_index = simulation_ordering.index(highlight_model)

    # Build temporary 1D arrays of parameters/offsets in correct ordering
    ordered_penalties = np.array([penalties[x] for x in simulation_ordering])

    limits = model_specification.parameter_limits

    ordered_parameters = [
        np.array(
            [
                model_parameters.model_parameters[x][parameter]
                for x in simulation_ordering
            ]
        )
        for index, parameter in zip(parameter_indices, use_parameters)
    ]

    bins = int(round(1.0 / visualisation_size))

    statistic = statistic if statistic is not None else "mean"

    # JB: I am 100% confident in this loop and that we are looping
    # over the correct axes. Do not change this loop.
    for parameter_y, axes_column in zip(parameter_indices, axes_grid):
        for parameter_x, ax in zip(parameter_indices, axes_column):
            limits_x = model_specification.parameter_limits[parameter_x]
            limits_y = model_specification.parameter_limits[parameter_y]
            name_x = model_specification.parameter_printable_names[parameter_x]
            name_y = model_specification.parameter_printable_names[parameter_y]

            is_center_line = parameter_x == parameter_y
            do_not_plot = is_center_line and remove_ticks

            if not do_not_plot:
                grid, xs, ys, _ = binned_statistic_2d(
                    x=ordered_parameters[parameter_x],
                    y=ordered_parameters[parameter_y],
                    values=ordered_penalties,
                    statistic=statistic,
                    bins=bins,
                )

                im = ax.pcolormesh(
                    xs,
                    ys,
                    grid.T,
                    norm=norm,
                    rasterized=True,
                )

                # Uncomment me if you don't believe the comment above
                # ax.text(0.5, 0.5, f"x={name_x}\ny={name_y}", transform=ax.transAxes, ha="center", va="center", color="white")

                if highlight_model is not None:
                    highlight_x = ordered_parameters[parameter_x][highlight_index]
                    highlight_y = ordered_parameters[parameter_y][highlight_index]

                    ax.scatter(
                        highlight_x,
                        highlight_y,
                        color="white",
                        edgecolor="black",
                    )

                ax.set_xlim(*limits_x)
                ax.set_ylim(*limits_y)

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

                if is_center_line:
                    ax.text(
                        0.5,
                        0.5,
                        f"{limits_x[0]:3.3f} <\n{name_x}\n< {limits_x[1]:3.3f}",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                    )
            else:
                ax.set_xlabel(name_x)
                ax.set_ylabel(name_y)

            # Set square in data reference frame
            ax.set_aspect(1.0 / ax.get_data_ratio())

    if use_colorbar:
        fig.colorbar(
            im,
            ax=axes_grid.ravel().tolist(),
            orientation="horizontal",
            label=f"{statistic.capitalize()} penalty along line of sight",
        )

    for a in axes_grid[:-1, :].flat:
        a.set_xlabel(None)
    for a in axes_grid[:, 1:].flat:
        a.set_ylabel(None)

    # As of matplotlib 3.3.4, with a large number of sub-plots this hangs...
    if grid_size > 4:
        fig.constrained_layout = False
        fig.subplots_adjust(0, 0, 1, 1, 0.005, 0.005)

    return fig, ax
