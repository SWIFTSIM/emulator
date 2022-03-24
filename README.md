SWIFT-Emulator
==============

[![Documentation Status](https://readthedocs.org/projects/swiftemulator/badge/?version=latest)](https://swiftemulator.readthedocs.io/en/latest/?badge=latest)
![Test Status](https://github.com/swiftsim/emulator/actions/workflows/pytest.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/swiftemulator.svg)](https://badge.fury.io/py/swiftemulator)
[![status](https://joss.theoj.org/papers/61d082196ef861cc0b612486c1fa6d40/status.svg)](https://joss.theoj.org/papers/61d082196ef861cc0b612486c1fa6d40)

The SWIFT-emulator (henceforth 'the emulator') was initially designed for [SWIFT](http://swift.dur.ac.uk)
outputs, and includes utilities to read and write SWIFT data.

The emulator can be used used to predict 
outputs of simulations without having to run them, by employing Gaussian Process
Regression with `george` and sensitivity analysis with `SALib`.

Dcumentation is available at [ReadTheDocs](https://swiftemulator.readthedocs.io/).

Predicting Simulations
----------------------

The emulator can predict a given scaling relation
(the relationship between two output variables in a simulation, for instance the
masses of galaxies and their size) when varying the underlying physical model
simulated in a continuous way.

As an example from cosmological simulations, imagine varing the energy that supernovae
release when they explode as a parameter `x`. This affects both the sizes and masses of galaxies.
The emulator, using a few 'base' simulations, performed with the real code,
at various values of `x` spaced evenly throughout the viable region, can predict
what the shape of the relationship between mass and size would be at other values
of `x`, given that it has been trained on the base simulation outputs.

Why SWIFT Emulator?
-------------------

The emulator works at a much higher level than other Gaussian Process Regression
libraries, such as `george` (which it is built on).

Working with base machine learning libraries can be tricky, as it typically
requries knowledge of how to structure input arrays in a specific way (both for
training and prediction). They also rarely also include model design routines.
Additionally, validation and visualisation routines
are typically missing.

The SWIFT Emulator package provides a one-stop solution, with a consistent API,
for developing a model design, running it (if using SWIFT), reading in data (
again if using SWIFT), building an emulation model, validating said model,
comparing the model against ground-truth data _across parameter space_ 
(e.g. observations), and visualising the results.

Installation
------------

The package can be installed easily from PyPI under the name `swiftemulator`,
so:

```
pip3 install swiftemulator
```

This will install all necessary dependencies.

The package can be installed from source, by cloning the repository and
then using `pip install -e .` for development purposes.


Requirements
------------

The package requires a number of numerical and experimental design packages.
These have been tested (and are continuously tested) using GitHub actions CI
to use the latest versions available on PyPI. See `requirements.txt` for
details for the packages required to develop SWIFT-Emulator. The packages
will be installed automatically by `pip` when installing from PyPI.


Authors
-------

+ Roi Kugel (@Moyoxkit)
+ Josh Borrow (@jborrow)
