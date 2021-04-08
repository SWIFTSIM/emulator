"""
Emulator generation object.
"""

import attr
import copy
import numpy as np

from typing import Hashable, List, Optional, Dict

from .model_parameters import ModelParameters
from .model_specification import ModelSpecification
from .model_values import ModelValues

from swiftemulator.emulators.gaussian_process import GaussianProcessEmulator
from swiftemulator.emulators.gaussian_process_mcmc import GaussianProcessEmulatorMCMC
from swiftemulator.emulators.gaussian_process_bins import GaussianProcessEmulatorBins
from swiftemulator.emulators.linear_model import LinearModelEmulator


@attr.s
class EmulatorGenerator(object):
    """
    Generator for emulators for individual scaling relations.

    Parameters
    ----------

    model_specification: ModelSpecification
        Full instance of the model specification.

    model_parameters: ModelParameters
        Full instance of the model parameters.

    Notes
    -----

    The required initialisation parameters are shared
    amongst all emulators that the emulator generator
    produces.
    """

    model_specification: ModelSpecification = attr.ib(
        validator=attr.validators.instance_of(ModelSpecification)
    )
    model_parameters: ModelParameters = attr.ib(
        validator=attr.validators.instance_of(ModelParameters)
    )

    def create_gaussian_process_emulator(
        self, model_values: ModelValues
    ) -> GaussianProcessEmulator:
        """
        Creates an individual emulator for an individual scaling
        relation described by the provided ``model_values``.

        Parameters
        ----------

        model_values, ModelValues
            The model values structure for this given scaling relation.
            This specifies the training data for the emulator.

        Returns
        -------

        emulator, GaussianProcessEmulator
            The built and trained emulator ready for prediction steps.
        """

        raise NotImplementedError(
            "This class has been deprecated. Please directly construct emualtors instead."
        )

    def create_gaussian_process_emulator_mcmc(
        self, model_values: ModelValues
    ) -> GaussianProcessEmulatorMCMC:
        """
        Creates the object needed for the hyperparameter_investigator
        function

        Parameters
        ----------

        model_values, ModelValues
            The model values structure for this given scaling relation.
            This specifies the training data for the emulator.

        Returns
        -------

        emulator, GaussianProcessEmulatorMCMC
            The built emulator ready for analysis of hyperparameters
        """

        raise NotImplementedError(
            "This class has been deprecated. Please directly construct emualtors instead."
        )

    def create_gaussian_process_emulator_bins(
        self, model_values: ModelValues
    ) -> GaussianProcessEmulatorBins:
        """
        Creates the object needed for the binned emulator

        Parameters
        ----------

        model_values, ModelValues
            The model values structure for this given scaling relation.
            This specifies the training data for the emulator.

        Returns
        -------

        emulator, GaussianProcessEmulatorMCMC
            The built emulator ready for analysis of hyperparameters
        """

        raise NotImplementedError(
            "This class has been deprecated. Please directly construct emualtors instead."
        )

    def create_linear_model_emulator(
        self, model_values: ModelValues
    ) -> LinearModelEmulator:
        """
        Creates an individual emulator for an individual scaling
        relation described by the provided ``model_values``.

        Parameters
        ----------

        model_values, ModelValues
            The model values structure for this given scaling relation.
            This specifies the training data for the emulator.

        Returns
        -------

        emulator, LinearModelEmulator
            The built and trained emulator ready for prediction steps.
        """

        raise NotImplementedError(
            "This class has been deprecated. Please directly construct emualtors instead."
        )
