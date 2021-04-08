"""
An example using the SWIFT pipeline i/o functions, testing how the stellar
mass function depends on the given parameters.

Requires the data available at:

``http://virgodb.cosma.dur.ac.uk/swift-webstorage/IOExamples/emulator_output.zip``

You will also need the observational data available at

``https://github.com/swiftsim/velociraptor-comparison-data``
"""

from swiftemulator.io.swift import load_parameter_files, load_pipeline_outputs
from swiftemulator.emulators.gaussian_process import GaussianProcessEmulator
from swiftemulator.mean_models import LinearMeanModel
from swiftemulator.mocking import mock_hypercube
from velociraptor.observations import load_observations
from swiftemulator.comparison import continuous_model_offset_from_observation
from swiftemulator.comparison.visualisation import visualise_offsets_mean

from glob import glob
from pathlib import Path
from tqdm import tqdm
from matplotlib.colors import Normalize

import matplotlib.pyplot as plt
import numpy as np
import corner

import os

files = [Path(x) for x in glob("./emulator_output/input_data/*.yml")]

filenames = {filename.stem: filename for filename in files}

spec, parameters = load_parameter_files(
    filenames=filenames,
    parameters=[
        "EAGLEFeedback:SNII_energy_fraction_min",
        "EAGLEFeedback:SNII_energy_fraction_max",
        "EAGLEFeedback:SNII_energy_fraction_n_Z",
        "EAGLEFeedback:SNII_energy_fraction_n_0_H_p_cm3",
        "EAGLEFeedback:SNII_energy_fraction_n_n",
        "EAGLEAGN:coupling_efficiency",
        "EAGLEAGN:viscous_alpha",
        "EAGLEAGN:AGN_delta_T_K",
    ],
    log_parameters=[
        "EAGLEAGN:AGN_delta_T_K",
        "EAGLEAGN:viscous_alpha",
        "EAGLEAGN:coupling_efficiency",
        "EAGLEFeedback:SNII_energy_fraction_n_0_H_p_cm3",
    ],
    parameter_printable_names=[
        "$f_{\\rm E, min}$",
        "$f_{\\rm E, max}$",
        "$n_{Z}$",
        "$\\log_{10}$ $n_{\\rm H, 0}$",
        "$n_{n}$",
        "$\\log_{10}$ $C_{\\rm eff}$",
        "$\\log_{10}$ $\\alpha_{\\rm V}$",
        "AGN $\\log_{10}$ $\\Delta T$",
    ],
)

value_files = [Path(x) for x in glob("./emulator_output/output_data/*.yml")]

filenames = {filename.stem: filename for filename in value_files}

values, units = load_pipeline_outputs(
    filenames=filenames,
    scaling_relations=["stellar_mass_function_100"],
    log_independent=["stellar_mass_function_100"],
    log_dependent=["stellar_mass_function_100"],
)

# Train an emulator for the space.

scaling_relation = values["stellar_mass_function_100"]
scaling_relation_units = units["stellar_mass_function_100"]

# We can now compare our real simulation to observational data!
observation = load_observations(
    "./observational_data/data/GalaxyStellarMassFunction/Vernon.hdf5"
)[0]

offsets = continuous_model_offset_from_observation(
    model_values=scaling_relation,
    observation=observation,
    unit_dict=scaling_relation_units,
    model_difference_range=[9.0, 11.0],
)

# Need to format the data for `corner` now.
fig, axes = visualise_offsets_mean(
    model_specification=spec,
    model_values=scaling_relation,
    model_parameters=parameters,
    offsets=offsets,
)

fig.tight_layout(h_pad=0.05, w_pad=0.05)
fig.savefig("real_simulation_corner_test.png")

train_model = lambda x: x.fit_model(
    model_specification=spec,
    model_parameters=parameters,
    model_values=scaling_relation,
)

emulator = GaussianProcessEmulator(
    model_specification=spec,
    model_parameters=parameters,
    model_values=scaling_relation,
)

emulator.build_arrays()
emulator.fit_model(mean_model=LinearMeanModel())

# Now that we have a trained model, we can super-sample that model
# into a much larger hyper-cube. This will allow us to more closely
# see which regions provide good results.

mock_values, mock_parameters = mock_hypercube(
    emulator=emulator, model_specification=spec, samples=512
)

# We can now compare our mocked simulation to observational data!
observation = load_observations(
    "./observational_data/data/GalaxyStellarMassFunction/Vernon.hdf5"
)[0]

offsets = continuous_model_offset_from_observation(
    model_values=mock_values,
    observation=observation,
    unit_dict=scaling_relation_units,
    model_difference_range=[9.0, 11.0],
)

# Need to format the data for `corner` now.

fig, axes = visualise_offsets_mean(
    model_specification=spec,
    model_values=mock_values,
    model_parameters=mock_parameters,
    offsets=offsets,
)

fig.tight_layout(h_pad=0.05, w_pad=0.05)
fig.savefig("mocked_simulation_corner_test.png")

# Finally, let's make a plot showing the 'best' model
# according to our metric!

best_key, best_value = min(offsets.items(), key=lambda x: x[1])
best_sim = mock_values.model_values[best_key]
best_sim_parameters = mock_parameters.model_parameters[best_key]
closest_true, _ = parameters.find_closest_model(
    mock_parameters.model_parameters[best_key]
)
closest_true = closest_true[0]
closest_true_values = scaling_relation.model_values[closest_true]

fig, ax = plt.subplots()

observation.plot_on_axes(ax)

ax.fill_between(
    10 ** closest_true_values["independent"],
    10 ** (closest_true_values["dependent"] - closest_true_values["dependent_error"]),
    10 ** (closest_true_values["dependent"] + closest_true_values["dependent_error"]),
    color="C1",
    alpha=0.5,
    linewidth=0.0,
)
ax.plot(
    10 ** closest_true_values["independent"],
    10 ** closest_true_values["dependent"],
    color="C1",
    label="Closest Real Model",
)

ax.fill_between(
    10 ** best_sim["independent"],
    10 ** (best_sim["dependent"] - best_sim["dependent_error"]),
    10 ** (best_sim["dependent"] + best_sim["dependent_error"]),
    color="C2",
    alpha=0.5,
    linewidth=0.0,
)
ax.plot(
    10 ** best_sim["independent"],
    10 ** best_sim["dependent"],
    color="C2",
    label="Predicted Best Model",
)

ax.loglog()

ax.legend()
ax.set_xlabel("Stellar Mass (100 kpc) [M$_\\odot$]")
ax.set_ylabel("Stellar Mass Function")

fig.tight_layout()
print(best_sim_parameters)
fig.savefig("best_model_comparison.png")