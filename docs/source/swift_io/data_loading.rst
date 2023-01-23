Loading SWIFT data
------------------

In order for this example to work you will need to
download some data from
`http://virgodb.cosma.dur.ac.uk/swift-webstorage/IOExamples/emulator_output.zip`.
This will contain a set of parameter files and a 
set of data files. The parameters files will be
used to retrieve the parameters of the Latin 
Hypercube, while the data files contain the
results for the scaling relations for each model.

It is adviced to use the SWIFT-io options if you
want to compare directly with observational data.
The main advantage being that loading the data in
this way will ensure that you use the correct units.

What will be required to load the data is a list 
of all the parameter files, and a list for all
the data files. This can easily be obtained using
:mod:`glob` and :mod:`Path`.

.. code-block:: python

    from glob import glob
    from pathlib import Path

    parameter_files = [Path(x) for x in glob("./emulator_output/input_data/*.yml")]
    parameter_filenames = {filename.stem: filename for filename in parameter_files}

    data_files = [Path(x) for x in glob("./emulator_output/output_data/*.yml")]
    data_filenames = {filename.stem: filename for filename in data_files}

For the parameter we use
:meth:`swiftemulator.io.swift.load_parameter_files`.
This reads in the parameters and returns both a `ModelSpecification`
and a `ModelParameters` container to pass to the emulator.

.. code-block:: python

    from swiftemulator.io.swift import load_parameter_files

    spec, parameters = load_parameter_files(
    filenames=parameter_filenames,
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

Just like for the experimental design, it is
important that the name used for `parameters`
is the same as the one used in the parameter 
file. Note also that you have to supply a list
of parameters that where sampled in log-space.
These are then trasformed to log-space before
being stored in a `ModelParameters` container.

To read the `ModelValues` the function 
:meth:`swiftemulator.io.swift.load_pipeline_outputs`
is used. In this case you have to supply the
filenames, and the name(s) of the scaling relation(s).
These names can be easily found in the data file 
and are set by the config used for the pipeline.
`log_independent` and `log_dependent` will cause
the x or y to be loaded in log-space.

.. code-block:: python

    from swiftemulator.io.swift import load_pipeline_outputs

    values, units = load_pipeline_outputs(
        filenames=data_filenames,
        scaling_relations=["stellar_mass_function_100"],
        log_independent=["stellar_mass_function_100"],
        log_dependent=["stellar_mass_function_100"],
    )

    scaling_relation = values["stellar_mass_function_100"]
    scaling_relation_units = units["stellar_mass_function_100"]

`load_pipeline_outputs` can return as many scaling
relations as required. `values` is dictionary that
contains a `ModelValues` container for each requested
scaling relation. A `ModelValues` container for a
single relation can be obtained by parsing it with
the correct name.

In addition to the basic dependent errors available,
we also provide the ability to use a histogram
through the parameter `histogram_for_error`, which returns
errors that are `1 / sqrt(N)` for all scaling relations loaded
in that same call.

At this point the data is loaded and you can build
and train your emulator.

.. code-block:: python

    from swiftemulator.emulators import gaussian_process

    emulator = gaussian_process.GaussianProcessEmulator()
    emulator.fit_model(model_specification=spec,
        model_parameters=parameters,
        model_values=scaling_relation,
    )