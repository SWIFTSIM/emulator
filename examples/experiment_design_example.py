"""
A basic SWIFT example simulation design.

The first parameter to the script is a base parameter file; if that is
not present then the only parameters we write are the ones varied here.

Produces thirty parameter files in a folder called experiment_design,
along with a corner plot called experiment_design.png.
"""

import sys
from pathlib import Path

try:
    base_parameter_file = Path(sys.argv[1])
except:
    base_parameter_file = None

from swiftemulator.design import latin
from swiftemulator.io.swift import write_parameter_files
from swiftemulator import ModelSpecification

number_of_simulations = 30
output_path = Path("experiment_design")
output_path.mkdir(exist_ok=True)

spec = ModelSpecification(
    number_of_parameters=5,
    parameter_names=[
        "EAGLEFeedback:SNII_energy_fraction_min",
        "EAGLEFeedback:SNII_energy_fraction_max",
        "EAGLEFeedback:SNII_energy_fraction_n_Z",
        "EAGLEFeedback:SNII_energy_fraction_n_0_H_p_cm3",
        "EAGLEFeedback:SNII_energy_fraction_n_n",
    ],
    parameter_printable_names=[
        "$f_{\\rm E, min}$",
        "$f_{\\rm E, max}$",
        "$n_{Z}$",
        "$\\log_{10}$ $n_{\\rm H, 0}$",
        "$n_{n}$",
    ],
    parameter_limits=[
        [0.0, 1.0],
        [1.0, 7.0],
        [-0.5, 5.0],
        [-1.0, 1.5],
        [-0.5, 5.0],
    ],
)

parameter_transforms = {"SNII_energy_fraction_n_0_H_p_cm3": lambda x: 10.0**x}

model_parameters = latin.create_hypercube(
    model_specification=spec,
    number_of_samples=number_of_simulations,
)

model_parameters.plot_model(
    model_specification=spec,
    filename=output_path / "experiment_design.png",
    corner_kwargs=dict(bins=5),
)

write_parameter_files(
    filenames={
        key: output_path / f"{key}.yml"
        for key in model_parameters.model_parameters.keys()
    },
    model_parameters=model_parameters,
    parameter_transforms=parameter_transforms,
    base_parameter_file=base_parameter_file,
)
