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
    emulator=emulator, model_specification=spec, samples=8192
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

corner_data = np.empty(
    (len(mock_parameters.model_parameters), spec.number_of_parameters),
    dtype=np.float32,
)
corner_weights = np.empty(len(mock_parameters.model_parameters), dtype=np.float32)

for index, (uid, model) in enumerate(mock_parameters.mock_parameters.items()):
    corner_data[index] = np.array(
        [model[parameter] for parameter in spec.parameter_names],
        dtype=np.float32,
    )

    corner_weights[index] = offsets[uid]

# Normalize and invert the corner weights, as we want to show 'better' models
# as 'brighter'
corner_weights = 1.0 - Normalize()(corner_weights)

corner.corner(
    xs=corner_data,
    bins=64,
    range=spec.parameter_limits,
    weights=corner_weights,
    labels=spec.parameter_printable_names,
)

plt.savefig("corner_test.png")