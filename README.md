SWIFT Emulator
==============

[![Documentation Status](https://readthedocs.org/projects/swiftemulator/badge/?version=latest)](https://swiftemulator.readthedocs.io/en/latest/?badge=latest)
![Test Status](https://github.com/swiftsim/emulator/actions/workflows/pytest.yml/badge.svg)

Emulator for SWIFT (http://swift.dur.ac.uk) outputs, used to predict 
outputs of simulations without having to run them. Employs Gaussian Process
Regression with `george` and sensitivity analysis with `SALib`.

Dcumentation is available at [ReadTheDocs](https://swiftemulator.readthedocs.io/).


Requirements
------------

The package requires a number of numerical and experimental design packages.
These have been tested (and are continuously tested) using GitHub actions CI
to use the latest versions available on PyPI. See `requirements.txt` for
details.


Authors
-------

+ Roi Kugel
+ Josh Borrow
