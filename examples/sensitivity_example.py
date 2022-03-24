"""
An example using the SWIFT i/o functions to demonstrate how a 
binwise sensitivity analysis is performed using the swift-emulator.

``http://virgodb.cosma.dur.ac.uk/swift-webstorage/IOExamples/emulator_output.zip``
"""

from swiftemulator.io.swift import load_parameter_files, load_pipeline_outputs
from glob import glob
from pathlib import Path
from swiftemulator.sensitivity.basic import (
    binwise_sensitivity,
    plot_binwise_sensitivity,
)

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

binwise = binwise_sensitivity(
    specification=spec,
    parameters=parameters,
    values=values["stellar_mass_function_100"],
)

fig, ax = plot_binwise_sensitivity(
    specification=spec,
    sensitivities=binwise,
    xlabel="Stellar Mass Bin [$\\log_{10}$ M$_\\odot$]",
    cbarlabel="Stellar Mass Function Response",
)

fig.savefig("sensitivity_analysis_smf.png")
