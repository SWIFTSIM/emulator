Emulator Options
================

For any smooth function with low dynamic range
the default gaussian process will provide all 
the accuracy that is necesarry for most purposes.
However, it is not hard to imagine certain
situations where taking some extra care before
invoking the GP can lead to more accuracy.
Here, some of the extra options that are
available for the emulation are highlighted.
The example data will be the Schecter function 
example:

.. code-block:: python

    import swiftemulator as se
    from swiftemulator.emulators import gaussian_process
    import numpy as np

    def log_schecter_function(log_M, log_M_star, alpha):
        M = 10 ** log_M
        M_star = 10 ** log_M_star
        return np.log10( (1 / M_star) * (M / M_star) ** alpha * np.exp(- M / M_star ))

    model_specification = se.ModelSpecification(
        number_of_parameters=2,
        parameter_names=["log_M_star","alpha"],
        parameter_limits=[[11.,12.],[-1.,-3.]],
        parameter_printable_names=["Mass at knee","Low mass slope"],
    )

    log_M_star = np.random.uniform(11., 12., 100)
    alpha      = np.random.uniform(-1., -3., 100)

    modelparameters = {}
    for unique_identifier in range(100):
        modelparameters[unique_identifier] = {"log_M_star": log_M_star[unique_identifier],
                                            "alpha": alpha[unique_identifier]}

    model_parameters = se.ModelParameters(model_parameters=modelparameters)

    modelvalues = {}
    for unique_identifier in range(100):
        independent = np.linspace(10,12,10)
        dependent = log_schecter_function(independent,
                                        log_M_star[unique_identifier],
                                        alpha[unique_identifier])
        dependent_error = 0.02 * dependent
        modelvalues[unique_identifier] = {"independent": independent,
                                        "dependent": dependent,
                                        "dependent_error": dependent_error}

    model_values = se.ModelValues(model_values=modelvalues)

    schecter_emulator = gaussian_process.GaussianProcessEmulator()
    schecter_emulator.fit_model(model_specification=model_specification,
                                model_parameters=model_parameters,
                                model_values=model_values)


Mean models
-----------

The most basic addition to a GP is to exchange
the constant (or zero) mean for a more complete
model. For the SWIFT-Emulator these can be found
under :meth:`swiftemulator.mean\_models`. All 
currently implement models come in the form of
different order polynomials. Much like the GP
you have to define your model first. The model can
then be passed to the GP which will fit the
coefficients and use that as a mean model.
The GP will then be used to model the residuals
between the polynomial fit and the data.

.. code-block:: python

    from swiftemulator.mean_models.polynomial import PolynomialMeanModel

    polynomial_model = PolynomialMeanModel(degree=1)

    schecter_emulator = gaussian_process.GaussianProcessEmulator(mean_model=polynomial_model)
    schecter_emulator.fit_model(model_specification=model_specification,
                            model_parameters=model_parameters,
                            model_values=model_values)

The polynomial model fits a polynomial surface
to all parameters. This includes not just the 
polynomial coefficients for each parameter but
also the linear combinations up to the degree
of the model. Be carefull picking a degree that
is very large, as it can quickly lead to 
over-fitting.

Emulating Bin-by-Bin
--------------------

The scaling relations obtained from simulations
are often binned relations. For the all-purpose
emulator we use the `independent` as an additional
parameter which the emulator uses for prediction,
but there are situation where you would prefer
modeling the response at each `independent` bin
seperately, instead of modeling it all at once.

When emulating bin-to-bin the main difference
is that your GP now comes from
:meth:`swiftemulator.emulators.gaussian_process_bins`.
Each bin will have a unique emulator, that is
trained on all data available for that bin.
A bin is created for each unique value of
the `independent` found in the `ModelValues`
container. It is extremely important that
each bin of the original data-set has exactly 
the same value for the `independent`. However,
the individual models do not need the same
sample of bins. If some models are missing
values for some of the bins, this is not
a problem.

Using the binned emulator is as simple as

.. code-block:: python

    from swiftemulator.emulators import gaussian_process_bins

    schecter_emulator_binned = gaussian_process_bins.GaussianProcessEmulatorBins()
    schecter_emulator_binned.fit_model(model_specification=model_specification,
                                       model_parameters=model_parameters,
                                       model_values=model_values)

Which has the same predicion functionality
as the standard `gaussian_process`.
Note that there is also a binned version
of the cross checks, 
:meth:`swiftemulator.sensitivity.cross\_check\_bins`,
which acts the same as the normal `cross_check`
but instead uses the binned emulator, making
it easy to compare the two methods.

1D Emulation
------------

Sometimes the emulation problem is better solved as

.. math::
    f(\vec\theta)

In this case we only have the model parameters.
the emulator won't be a function of an additional x
parameter stored in the model values. In this case the
use can use :meth:`swiftemulator.emulators.gaussian_process_one_dim`.
This method has similar functionality as the other
emulator types. It will still need a ModelValues
container. Here is an example of how such a container
should look like:

.. code-block:: python

    modelvalues = {}
    for unique_identifier in range(100):
        dependent = func(a_arr[unique_identifier], b_arr[unique_identifier])
        dependent_error = 0.02 * dependent
        modelvalues[unique_identifier] = {"independent": [None],
                                          "dependent": [dependent],
                                        "dependent_error": [dependent_error]}

In order to make use of the general emulator
containers, it is still required to provide the values
as list. In this case the lists will only contain a single
value. The independent value will not be read. When your
data is in the correct format the emulator can be trained
like all the other methods.

.. code-block:: python

    from swiftemulator.emulators import gaussian_process_one_dim

    schecter_emulator_one_dim = gaussian_process_one_dim.GaussianProcessEmulator1D()
    schecter_emulator_one_dim.fit_model(model_specification=model_specification,
                                       model_parameters=model_parameters,
                                       model_values=model_values)

The only other thing of note is that while
`predict_values` retains the same functionality,
you are no longer required to specify any independent
values. The prediction is now based purely of the
given values of the model parameters.