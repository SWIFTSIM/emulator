Welcome to SWIFT Emulator's Documentation
=========================================

The SWIFT-Emulator is a python toolkit for using Gaussian
Process machine learning to produce synthetic simulation data
by interpolating between base outputs. It excels at creating
synthetic scaling relations across large swathes of model
parameter space, as it was created to model galaxy scaling
relations as a function of galaxy formation model parameters
for calibration purposes.

It includes functionality to:

- Generate parameters to perform ground truth runs with in an
  efficient way as a latin hypercube.
- Train machine learning models, including linear models and
  Gaussian Process Regression models (with mean models), on
  this data in a very clean way.
- Generate densly populated synthetic data across the original
  parameter space, and tools to generate complex model
  discrepancy descriptions (known here as penalty functions).
- Generate sweeps across model parameter space for the emulated
  scaling relations to assist in physical insight, as well as
  sensitivity analysis tools based upon raw and synthetic data.
- Visualise the resulting penalty data to assist in model choice
  decisions.
- Produce inputs and read outputs from the cosmological code
  SWIFT that processed by VELOCIraptor and the swift-pipeline.

Information about `SWIFT` can be found 
`here <http://swift.dur.ac.uk/>`_, Information about 
`VELOCIraptor` can be found 
`here <https://velociraptor-stf.readthedocs.io/en/latest/>`_
and tnformation about the `SWIFT-pipeline` can be found 
`here <https://github.com/SWIFTSIM/pipeline>`_.

By combining a selection of SWIFT-io and GP analysis 
tools, the SWIFT-Emulator serves to make emulation of 
SWIFT outputs very easy, while staying flexible enough 
to emulate anything, given a good set of training data.

.. toctree::
   :maxdepth: 2

   getting_started/index
   
   emulator_analysis/index

   emulator_options/index
   
   swift_io/index

   comparisons/index

   modules/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
