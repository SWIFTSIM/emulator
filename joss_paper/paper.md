---
title: 'swift-emulator: A Python package for emulation of simulated scaling relations'
tags:
  - Python
  - astronomy
  - Simulations
  - Cosmology
  - Machine Learning
authors:
  - name: Roi Kugel^[co-first author] # note this makes a footnote saying 'co-first author'
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Josh Borrow^[co-first author] # note this makes a footnote saying 'co-first author'
    affiliation: 2
affiliations:
 - name: Leiden Observatory, Leiden University
   index: 1
 - name: Institution Name
   index: 2
date: 13 August 2017
bibliography: paper.bib

---

# Summary

The forces on stars, galaxies, and dark matter under external gravitational
fields lead to the dynamical evolution of structures in the universe. The orbits
of these bodies are therefore key to understanding the formation, history, and
future state of galaxies. The field of "galactic dynamics," which aims to model
the gravitating components of galaxies to study their structure and evolution,
is now well-established, commonly taught, and frequently used in astronomy.
Aside from toy problems and demonstrations, the majority of problems require
efficient numerical tools, many of which require the same base code (e.g., for
performing numerical orbit integration).

# Background

One of the limits of doing cosmological (hydrodynamical) simulations is
that any simulation is limited to only a single set of parameters. Cosmological
simulations can also be computationally quite expensive, with the cheapest runs
already costing hundreds of cpu hours. This makes the use of cosmological
simulations in state of the of the art fitting pipelines computationally
unfeasable. In order to get a statistical grip on the models of cosmology
and galaxy formation, a better solution is needed.

This problem is also a major limiting factor in "calibration" of the
sub-resolution (subgrid) models that are often used. Works like (CITE CRAIN EAGLE
AND BAHAMAS 2017) are able to "match" observed relations by eye, but a
statistical ground for the chosen parameters is missing. This poses
a signifcant problem for cosmology, where a deeper understanding of
our subgrid models will be required to understand upcoming surveys
like LSST and EUCLID.

A solution here comes trough the use of machine learning techniques. By
training emulators on a limited amount of simulations. This leads to having
a fully continuous model based on the variations. Instead of sampling a new
simulation for each required datapoint, the emulator can predict the results a
simulation would give for that set of parameters. This makes it feasable
to use methods like MCMC based purely of simulation results.

# Emulator Requirements

For emulation in hydro simulations we are most interested in emulating
scaling relations of the following form:

$$f(x,\vec\theta)$$

For each simulation we will have a relation where we calculate some
dependent quantity as a function of another indepdent quantity $x$.
Depending on the type of binning used the sampling in $x$ will be
roughly similar between simulations. $\vec\theta$ represent the
values of the parameters that can only be varied from simulation
to simulation. To give an example, one of the things that could
be emulated could be the the mass fraction of gas in clusters. The
gas fraction as a function of cluster mass can be meassured
from each individual simulation. In this case a simulation parameter,
like the energy from supernovas, would be part of the $\theta$
vector. Using emulators we want to be able to see how the full gas fraction vs
cluster mass relation changes as a function of parameters like
the supernova strength. This distinction between the values obtained from
the simulation and the parameters of the simulation model is absent
when setting up the emulator. And setting up the training data
correctly can therefore pose a significant challenge.

There is also the issue of having to get a correct sampling for
$\theta$. In order to save computational time, it is important
to have an efficient sampling of the parameter space. This includes
not just the sampling itself, but also the possibility to sample
parameters in transformed coordinates (like in logarithmic space).

Once the emulator is working it can be challenging to make
some of the standard tests of how the emulator is performing.
Things like cross-checks or parameter sweeps have to be implemented
by hand, making proper use of emulators more difficult.

# Why swift-emulator?

Many packages cexist for Gausian process emulation, like
`george` (CITE) (which provides the basis for `swift-emulator`),
`gpytorch` (CITE) `GPy` (CITE). Additionally, a package like
`pyDOE` (CITE) can be used to set up efficient parameter samplings.
However, most of these packages operate close to theory, and create
a significant barrier for entry.

With `swift-emulator` we aim to provide a single `python` package
that interfaces with available tools at a high level. Additionaly
we aim to streamline the processes by providing i/o tools for the
SWIFT simulation code (CITE). This is done in a modular
fashion, giving the users the freedom to change any steps along the way.
`swift-emulator` provides many methods that work out of the box,
removing the barrier to entry, and aim at making emulator methods easy to
use. The more wide-spread use of emulators will boost the potential of 
future simulation projects.

`swift-emulator` combines these tools to streamline the complete emulation
process. There are tools for experimental deisgn. For simulations done with
SWIFT, parameter files can be created and simulation outputs can be loaded
in. The results can then be used to train an emulator that can make 
predictions for the scaling relations in the simulation. There are also methods
to do cross-checks to find the accuracy of the emulator. And there is a
simple method to do a parameter sweep, to see what the impact of simulation
parameters is on a given scaling relation. Finally there are tools for comparing
the emulated relations with other data.

`swift-emulator` is currently being used for two of the flagship simulation
projects done using the SWIFT simulation code, ranging accros five orders of 
magnitude in mass resolution. The package is being used to allow modern
simulations to reporduce key observations with high accuracy.

Finally `swift-emulator` has many options to optimise the methods for
specific emulation problems. While the focus so far has been on integration
with SWIFT, it is possible to add i/o with other simulation codes.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References