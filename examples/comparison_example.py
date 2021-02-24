"""
An example using some SWIFT data to show comparisons between the
mass function and some data.

Requires the data available at:

``http://virgodb.cosma.dur.ac.uk/swift-webstorage/IOExamples/emulator_output.zip``

You will also need the observational data at

``https://github.com/swiftsim/velociraptor-comparison-data/``
"""

from swiftemulator.io.swift import load_pipeline_outputs
from swiftemulator.comparison import continuous_model_offset_from_observation
from velociraptor.observations import load_observations
from glob import glob
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

import os

value_files = [Path(x) for x in glob("./emulator_output/output_data/*.yml")]

filenames = {filename.stem: filename for filename in value_files}

values, units = load_pipeline_outputs(
    filenames=filenames,
    scaling_relations=["stellar_mass_function_100"],
    log_independent=["stellar_mass_function_100"],
    log_dependent=["stellar_mass_function_100"],
)

observation = load_observations(
    "./observational_data/data/GalaxyStellarMassFunction/Vernon.hdf5"
)[0]

# Calculate the offsets

offsets = continuous_model_offset_from_observation(
    model_values=values["stellar_mass_function_100"],
    observation=observation,
    unit_dict=units["stellar_mass_function_100"],
    model_difference_range=[9.0, 11.0],
)

# Lowest offset is best fit, highest offset is worst fit.

offset_ids = list(offsets.keys())
offset_values = np.array([offsets[k] for k in offset_ids])

best_fit = offset_ids[offset_values.argmin()]
worst_fit = offset_ids[offset_values.argmax()]

fig, ax = plt.subplots(constrained_layout=True)

observation.plot_on_axes(axes=ax)

plot_func = lambda ax, id, l: ax.plot(
    10 ** values["stellar_mass_function_100"].model_values[id]["independent"],
    10 ** values["stellar_mass_function_100"].model_values[id]["dependent"],
    label=l,
)

plot_func(ax, best_fit, "Best Fit")
plot_func(ax, worst_fit, "Worst Fit")

ax.set_xlabel("Stellar Mass (100 kpc) [M$_\\odot$]")
ax.set_ylabel("Stellar Mass Function")

ax.legend()
ax.loglog()

fig.savefig("stellar_mass_function_offset_comparison.png")
