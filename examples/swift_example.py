"""
An example using the SWIFT pipeline i/o functions, testing what occurs when
a single simulation is left out and then emulated.

Requires the data available at:

``http://http://virgodb.cosma.dur.ac.uk/swift-webstorage/IOExamples/emulator_output.zip``
"""

from swiftemulator.io.swift import load_parameter_files, load_pipeline_outputs
from swiftemulator.emulators.gaussian_process import GaussianProcessEmulator
from glob import glob
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

import os

files = [Path(x) for x in glob("./emulator_data/input_data/*.yml")]

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
        "$n_{\\rm H, 0}$",
        "$n_{n}$",
        "$C_{\\rm eff}$",
        "$\\alpha_{\\rm V}$",
        "AGN $\\Delta T",
    ],
)

value_files = [Path(x) for x in glob("./emulator_data/output_data/*.yml")]

filenames = {filename.stem: filename for filename in value_files}

values, units = load_pipeline_outputs(
    filenames=filenames,
    scaling_relations=["stellar_mass_function_100"],
    log_independent=["stellar_mass_function_100"],
    log_dependent=["stellar_mass_function_100"],
)


# Now let's try to leave one out one at a time

leave_out_order = list(filenames.keys())
emulators = {k: None for k in leave_out_order}
scaling_relation = values["stellar_mass_function_100"]

for unique_identifier in tqdm(leave_out_order):
    left_out_data = scaling_relation.model_values.pop(unique_identifier)

    emulator = GaussianProcessEmulator(
        model_specification=spec,
        model_parameters=parameters,
        model_values=scaling_relation,
    )

    emulator.build_arrays()

    emulators[unique_identifier] = emulator

    scaling_relation.model_values[unique_identifier] = left_out_data


train_model = lambda x: x.fit_model(fit_linear_model=True)

list(map(train_model, [emulators["9"]]))


try:
    os.mkdir("leave_one_out_figures")
except:
    pass


emulate_at = np.linspace(7, 12, 100)

for unique_identifier in emulators.keys():
    fig, ax = plt.subplots(constrained_layout=True)

    emulated, emulated_error = emulators[unique_identifier].predict_values(
        emulate_at, model_parameters=parameters.model_parameters[unique_identifier]
    )

    ax.fill_between(
        emulate_at,
        emulated - np.sqrt(emulated_error),
        emulated + np.sqrt(emulated_error),
        color="C1",
        alpha=0.3,
        linewidth=0.0,
    )

    ax.errorbar(
        scaling_relation.model_values[unique_identifier]["independent"],
        scaling_relation.model_values[unique_identifier]["dependent"],
        yerr=scaling_relation.model_values[unique_identifier]["dependent_error"],
        label="True",
        marker=".",
        linestyle="none",
        color="C0",
    )

    ax.plot(emulate_at, emulated, label="Emulated", color="C1")

    plt.xlabel("Log($M_*$ / M$_\odot$)")
    plt.ylabel("Log(Mass Function)")
    plt.legend()
    plt.title(f"Leave Out Run {unique_identifier}")

    plt.savefig(f"leave_one_out_figures/leave_out_{unique_identifier}.png", dpi=300)

