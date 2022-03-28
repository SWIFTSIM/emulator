Experimental Design
-------------------

One part of SWIFT-Emulator's i/o features is
the ability to generate Latin Hypercube (LH)
designs and save them to SWIFT parameter files,
one file for each set of parameters. For this
example you will need to download some data
from `http://virgodb.cosma.dur.ac.uk/swift-webstorage/IOExamples/emulator_output.zip`.

We do this by combining the :meth:`swiftemulator.design`
with :meth:`swiftemulator.io.swift`. First we have 
to specify what parameter we want to vary.

.. code-block:: python

    from swiftemulator.design import latin
    from swiftemulator.io.swift import write_parameter_files
    from swiftemulator import ModelSpecification

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

    parameter_transforms = {"SNII_energy_fraction_n_0_H_p_cm3": lambda x: 10.0 ** x}

In this case it is important that your
`parameter_names` are identical to the
names in the SWIFT parameter file. The parameter
file is a `.yml` file so the individual parameters
should be named in that format. 

In this case the fourth parameter, 
`EAGLEFeedback:SNII_energy_fraction_n_0_H_p_cm3`
is sampled in log-space. The `parameter_limits`
are given in log-space as well in this case, but
you need to define the transformation needed when
going from the design space, to the value you
want to put in the parameter file.

Generating the LH can then be done with
:meth:`swiftemulator.design.latin.create\_hypercube`.

.. code-block:: python

    number_of_simulations = 30

    model_parameters = latin.create_hypercube(
        model_specification=spec,
        number_of_samples=number_of_simulations,
    )

Now we can use the SWIFT i/o to write these
to a set of parameter files. You will have
noticed that we only need to provide the
parameters that we want to vary. This is 
because we provide `write_parameter_files`
with a base parameter file. This file
should hold the base values for all
other parameters. Example parameter files
can be found on the main `SWIFT` repository.
For this example we will use one of the
parameter files from the example hypercube.

.. code-block:: python

    base_parameter_file = "emulator_output/input_data/1.yml"
    output_path = "."

    write_parameter_files(
    filenames={
        key: f"{output_path} / {key}.yml"
        for key in model_parameters.model_parameters.keys()
    },
    model_parameters=model_parameters,
    parameter_transforms=parameter_transforms,
    base_parameter_file=base_parameter_file,
    )

This writes 30 files to the current
directory. These files can then be used to run
SWIFT for each of the models.