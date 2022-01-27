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