SWIFT Emulator
==============

Emulator for SWIFT (http://swift.dur.ac.uk) outputs, used to predict 
outputs of simulations without having to run them. Employs Gaussian Process
 Regression with `george` and sensitivity analysis with `SALib`.

Dcumentation is available at [ReadTheDocs](https://swiftemulator.readthedocs.io/).

Authors:

+ Roi Kugel
+ Josh Borrow


Requirements
------------

The package requires a number of numerical and experimental design packages.
These have been tested (and are continuously tested) using GitHub actions CI
to use the latest versions available on PyPI. See `requirements.txt` for
details.