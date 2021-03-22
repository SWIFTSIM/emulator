"""
Visualisation functions for comparison datasets.

Allows you to project a plausibility region for each
parameter cross-correlation.
"""

from swiftemulator.backend.model_parameters import ModelParameters
from swiftemulator.backend.model_specification import ModelSpecification
from swiftemulator.backend.model_values import ModelValues

from typing import Dict, Hashable, Tuple, Iterable
from matplotlib.colors import Normalize
from swiftsimio.visualisation.projection import scatter
from scipy.stats import binned_statistic_2d

import matplotlib.pyplot as plt
import numpy as np


def visualise_offsets_mean(
    model_specification: ModelSpecification,
    model_values: ModelValues,
    model_parameters: ModelParameters,
    offsets: Dict[Hashable, float],
    vmin: float = 0.5,
    vmax: float = 0.5,
    remove_ticks: bool = False,
) -> Tuple[plt.Figure, Iterable[plt.Axes]]:
    number_of_parameters = model_specification.number_of_parameters
    grid_size = number_of_parameters

    fig, axes_grid = plt.subplots(
        grid_size,
        grid_size,
        figsize=(grid_size, grid_size),
        squeeze=True,
        sharex="col",
        sharey="row",
    )

    visualisation_size = 2.0 / np.sqrt(len(model_values.model_values))
    simulation_ordering = list(model_values.model_values.keys())

    # Build temporary 1D arrays of parameters/offsets in correct ordering
    ordered_offsets = np.array([offsets[x] for x in simulation_ordering])
    ordered_norms = 1.0 - ordered_offsets

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

    smoothing_lengths = np.ones_like(ordered_offsets) * visualisation_size

    for parameter_x, axes_column in enumerate(axes_grid):
        for parameter_y, ax in enumerate(axes_column):
            limits_x = model_specification.parameter_limits[parameter_x]
            limits_y = model_specification.parameter_limits[parameter_y]
            name_x = model_specification.parameter_printable_names[parameter_x]
            name_y = model_specification.parameter_printable_names[parameter_y]

            norm_grid = scatter(
                x=ordered_parameters[parameter_x],
                y=ordered_parameters[parameter_y],
                m=np.ones_like(ordered_norms),
                h=smoothing_lengths,
                res=512,
            )

            weighted_grid = scatter(
                x=ordered_parameters[parameter_x],
                y=ordered_parameters[parameter_y],
                m=ordered_norms,
                h=smoothing_lengths,
                res=512,
            )

            norm_grid[norm_grid == 0.0] = 1.0

            ratio_grid = weighted_grid / norm_grid

            im = ax.imshow(
                ratio_grid,
                extent=limits_y + limits_x,
                origin="lower",
                norm=Normalize(vmin=vmin, vmax=vmax, clip=True),
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


def visualise_offsets_generic_statistic(
    model_specification: ModelSpecification,
    model_values: ModelValues,
    model_parameters: ModelParameters,
    offsets: Dict[Hashable, float],
    statistic: Optional[str] = None,
    vmin: float = 0.5,
    vmax: float = 0.5,
    remove_ticks: bool = False,
) -> Tuple[plt.Figure, Iterable[plt.Axes]]:
    number_of_parameters = model_specification.number_of_parameters
    grid_size = number_of_parameters

    fig, axes_grid = plt.subplots(
        grid_size,
        grid_size,
        figsize=(grid_size, grid_size),
        squeeze=True,
        sharex="col",
        sharey="row",
    )

    visualisation_size = 4.0 / np.sqrt(len(model_values.model_values))
    simulation_ordering = list(model_values.model_values.keys())

    # Build temporary 1D arrays of parameters/offsets in correct ordering
    ordered_offsets = np.array([offsets[x] for x in simulation_ordering])
    ordered_norms = 1.0 - ordered_offsets

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

    bins = np.linspace(0.0, 1.0, 1.0 / visualisation_size)

    statistic = statistic if statistic is not None else "mean"

    for parameter_x, axes_column in enumerate(axes_grid):
        for parameter_y, ax in enumerate(axes_column):
            limits_x = model_specification.parameter_limits[parameter_x]
            limits_y = model_specification.parameter_limits[parameter_y]
            name_x = model_specification.parameter_printable_names[parameter_x]
            name_y = model_specification.parameter_printable_names[parameter_y]

            grid = binned_statistic_2d(
                x=ordered_parameters[name_x],
                y=ordered_parameters[name_y],
                values=ordered_norms,
                statistic=statistic,
                bins=bins,
            )

            im = ax.imshow(
                grid,
                extent=limits_y + limits_x,
                origin="lower",
                norm=Normalize(vmin=vmin, vmax=vmax, clip=True),
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