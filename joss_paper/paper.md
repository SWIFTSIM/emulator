---
title: 'swift-emulator: A Python package for emulation of simulated scaling relations'
tags:
  - Python
  - astronomy
  - dynamics
  - galactic dynamics
  - milky way
authors:
  - name: Adrian M. Price-Whelan^[co-first author] # note this makes a footnote saying 'co-first author'
    orcid: 0000-0000-0000-0000
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID^[co-first author] # note this makes a footnote saying 'co-first author'
    affiliation: 2
  - name: Author with no affiliation^[corresponding author]
    affiliation: 3
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University
   index: 1
 - name: Institution Name
   index: 2
 - name: Independent Researcher
   index: 3
date: 13 August 2017
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
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

One of the main limits of doing cosmological (hydrodynamic) simulations is
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
our subgrid models will be required to understand upcoming surveys (CITE MARCEL)

A solution here comes trough the use of machine learning techniques. By
training emulators on a limited amount of simulations. This leads to having
a fully continuous model based on the variations. Instead of sampling a new
simulation for each required datapoints, the emulator can predict how a
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
to simulation. This distinction between the values obtained from
the simulation and the parameters of the simulation model is absent
when setting up the emulator. And setting up the training data
correctly can therefore pose a significant challenge.

There is also the issue of having to get a correct sampling for
$\theta$. In order to save computational time, it is important
to have an efficient sampling of the parameter space. This includes
not just the sampling itself, but also the possibility to sample
parameters in transformed coordinates (like logspace).

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
fashion, giving the user the freedom to change any steps along the way.
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
parameters is on the scaling relation. Finally there are tools for comparing
with data, and finding models that fit well to that data. 

`swift-emulator` is currently being used for two of the flagship simulation
projects done using the SWIFT simulation code, ranging acros five orders of 
magnitude in mass resolution. The package is being used to allow modern
simulations to reporduce key observations with high accuracy.

Finally `swift-emulator` has many options to optimise the methods for
specific emulation problems. While the focus so far has been on integration
with SWIFT, it is possible to add i/o with other simulation codes.



# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

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

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References