"""
Contains functions for converting model values containers to
errors.
"""

import numpy as np

from scipy.interpolate import interp1d

from swiftemulator import ModelValues
from typing import Optional


def convert_dependent_error_to_standard_error(
    scaling_relation: ModelValues,
    histogram: ModelValues,
    log_independent: bool,
    log_dependent: bool,
    correction_factor: Optional[float] = 1.253,
    interpolation: Optional[str] = None,
) -> ModelValues:
    """
    Takes a model values container, along with another model
    values container with a histogram, and converts the
    ``dependent_error`` in the scaling relation to be
    standard errors. The assumption is that ``scaling_relation``
    contains a median line, hence the ``correction factor`` of
    1.253. If you have a mean line, you will need to change
    ``correction_factor`` to 1.0. The histogram is interpolated
    to the bin centers of the scaling relation.

    Parameters
    ----------

    scaling_relation: ModelValues
        The scaling relation to apply standard errors to. A copy
        will be created, and this will not be modified.

    histogram: ModelValues
        A model values container for the histogram. Note that this
        must have the same independent variable as ``scaling_relation``,
        including whether it has been logarithmically scaled, but
        must have a linear dependent variable (i.e. histogram(x) returns
        the number of items at that independent variable x). The unique
        identifiers must match the scaling relation.

    log_independent: bool
        Is the independent variable of both the histogram and scaling
        relation logarithmically scaled?

    log_dependent: bool
        Is the dependent variable of the scaling relation logarithmically
        scaled? The histogram must be linear along the dependent variable.

    correction_factor: float, optional
        The correction factor on the standard error. For a median this
        is 1.253 (the default); for a mean this is 1.0.

    interpolation: str, optional
        Interpolation style to pass to ``scipy.interpolate.interp1d``. By
        default we use linear interpolation.


    Returns
    -------

    modified_model_values: ModelValues
        The modified model values container with the ``dependent_error``
        now set as standard error on the median (or mean, if you changed
        the correction factor).


    Notes
    -----

    The correction for the median standard error only works in a well
    sampled gaussian distribution. This is probably not true, and poorly
    sampled distributions are actually more sensitive to skew (and hence
    the error in the correction factor is larger) than well sampled
    distribtuions. See ``https://davidmlane.com/hyperstat/A106993.html``.

    Additionally, this method assumes that your bins are of roughly the
    same size as the histogram bins. As such this is simply a rough estimate
    of the standard error and is not a replacement for fully calculated
    standard errors on the fly. This is also only acceptable for 'rescuing'
    datasets that do not have their bin edges saved; so if you are loading
    SWIFT data from the pipeline, you should use the options in the
    SWIFT i/o to use the sampling errors instead.

    To avoid divide by zero errors, when zero items are found in the histogram,
    the standard errors are treated as equal to the value (i.e. the number of
    items in the bin is artificially set to 1.0).
    """

    unique_identifiers = set(scaling_relation.keys())
    unique_identifiers_histogram = set(histogram.keys())

    if unique_identifiers != unique_identifiers_histogram:
        raise ValueError(
            "The unique identifiers for the scaling relation and histogram do not match."
        )

    # Interpolate the histograms to the bin centers of the
    # real model values container. The histograms must not
    # be logarithmically scaled when loading in!

    interpolation_style = interpolation if interpolation is not None else "linear"

    modified_values = {}

    for uid in unique_identifiers:
        relation = scaling_relation[uid]
        hist = histogram[uid]

        interpolated = interp1d(
            x=hist["independent"],
            y=hist["dependent"],
            kind=interpolation_style,
            copy=False,
            bounds_error=False,
            fill_value="extrapolate",
        )

        num_items_in_bins = interpolated(relation["independent"])
        num_items_in_bins[num_items_in_bins == 0.0] = 1.0

        standard_factor = correction_factor / np.sqrt(num_items_in_bins)

        if log_dependent:
            # In log space, up and down errorbars have different sizes!
            dependent_log = relation["dependent"]
            dependent_no_log = np.power(10.0, dependent_log)

            standard_error_size = dependent_no_log * standard_factor

            standard_error = np.array(
                [
                    dependent_log - np.log10(dependent_no_log - standard_error_size),
                    np.log10(dependent_no_log + standard_error_size) - dependent_log,
                ]
            )
        else:
            standard_error = relation["dependent"] * standard_factor

        modified_values[uid] = dict(
            independent=relation["independent"],
            dependent=relation["dependent"],
            dependent_error=standard_error,
        )

    return ModelValues(modified_values)


def convert_dependent_error_to_standard_error_using_edges(
    scaling_relation: np.array,
    scaling_relation_bin_edges: np.array,
    histogram: np.array,
    histogram_bin_edges: np.array,
    log_independent: bool,
    log_dependent: bool,
    correction_factor: Optional[float] = 1.253,
    interpolation: Optional[str] = None,
) -> np.array:
    """
    Takes a raw scaling relation, specified as arrays, and uses an associated
    histogram to generate sampling errors. The histogram is converted to
    cumulative, interpoltaed, and used to determine the number of objects
    used to sample each bin giving standard errors.

    Parameters
    ----------

    scaling_relation: np.array
        The scaling relation dependent variables.

    scaling_relation_bin_edges: np.array
        Scaling relation bin edges (should be one longer than the scaling
        relation dependent variables).

    histogram: np.array
        Histogram values, must be linearly scaled and _not_ cumulative.
        This should specify the number of objects between the associated
        ``histogram_bin_edges``.

    histogram_bin_edges: np.array
        Histogram bin edges (should be one longer than the histogram).

    log_independent: bool
        Is the independent variable of both the histogram and scaling
        relation logarithmically scaled?

    log_dependent: bool
        Is the dependent variable of the scaling relation logarithmically
        scaled? The histogram must be linear along the dependent variable.

    correction_factor: float, optional
        The correction factor on the standard error. For a median this
        is 1.253 (the default); for a mean this is 1.0.

    interpolation: str, optional
        Interpolation style to pass to ``scipy.interpolate.interp1d``. By
        default we use linear interpolation.


    Returns
    -------

    standard_error: np.array
        The standard error on the scaling relation in those bins.


    Notes
    -----

    To avoid divide by zero errors, when zero items are found in the histogram,
    the standard errors are treated as equal to the value (i.e. the number of
    items in the bin is artificially set to 1.0).
    """

    if len(histogram) != len(histogram_bin_edges) - 1:
        raise ValueError(
            "Histogram does not have correct number of bins "
            f"{len(histogram)} / {len(histogram_bin_edges)}."
        )

    if len(scaling_relation) != len(scaling_relation_bin_edges) - 1:
        raise ValueError(
            "Scaling relation does not have correct number of bins "
            f"{len(scaling_relation)} / {len(scaling_relation_bin_edges)}."
        )

    # Convert the histogram to a reverse cumulatively summed histogram.
    cumulative_histogram = np.cumsum(histogram[::-1])[::-1]
    # These values are actually the true values at the left hand bin edges; let's
    # add a zero to the RHS as there are no known objects beyond that value. There
    # is an implicit cast on the zero here but let's just hope that works.
    cumulative_histogram = np.append(cumulative_histogram, 0)

    interpolation_style = interpolation if interpolation is not None else "linear"

    interpolated_histogram = interp1d(
        x=histogram_bin_edges,
        y=cumulative_histogram,
        kind=interpolation_style,
        copy=False,
        bounds_error=False,
        fill_value=(cumulative_histogram[0], cumulative_histogram[-1]),
    )

    scaling_relation_edge_num_items = interpolated_histogram(scaling_relation_bin_edges)

    num_items_in_bins = -1.0 * np.diff(scaling_relation_edge_num_items)
    num_items_in_bins[num_items_in_bins == 0.0] = 1.0

    standard_factor = correction_factor / np.sqrt(num_items_in_bins)

    if log_dependent:
        # In log space, up and down errorbars have different sizes!
        dependent_log = scaling_relation
        dependent_no_log = np.power(10.0, dependent_log)

        standard_error_size = dependent_no_log * standard_factor

        standard_error = np.array(
            [
                dependent_log - np.log10(dependent_no_log - standard_error_size),
                np.log10(dependent_no_log + standard_error_size) - dependent_log,
            ]
        )
    else:
        standard_error = scaling_relation * standard_factor

    return standard_error
