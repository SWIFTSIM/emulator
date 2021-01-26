"""
Emulator generation object.
"""

import attr
import numpy as np

from typing import Hashable, List, Optional

from .model_parameters import ModelParameters
from .model_specification import ModelSpecification
from .model_values import ModelValues


@attr.s
class GaussianProcessEmulator(object):
    """
    Generator for emulators for individual scaling relations.

    Parameters
    ----------

    model_specification: ModelSpecification
        Full instance of the model specification.

    model_parameters: ModelParameters
        Full instance of the model parameters.

    model_values: ModelValues
        Full instance of the model values describing
        this individual scaling relation.
    """

    model_specification: ModelSpecification = attr.ib(
        validator=attr.validators.instance_of(ModelSpecification)
    )
    model_parameters: ModelParameters = attr.ib(
        validator=attr.validators.instance_of(ModelParameters)
    )
    model_values: ModelValues = attr.ib(
        validator=attr.validators.instance_of(ModelValues)
    )

    ordering: Optional[List[Hashable]] = None
    independent_variables: Optional[np.array] = None
    dependent_variables: Optional[np.array] = None
    dependent_variable_errors: Optional[np.array] = None

    def build_arrays(self) -> np.ndarray:
        """
        Builds the arrays for passing to `george`.
        """

        model_values = self.model_values.model_values
        unique_identifiers = model_values.keys()
        number_of_independents = self.model_values.number_of_variables
        number_of_model_parameters = self.model_specification.number_of_parameters
        model_parameters = self.model_parameters.model_parameters

        parameter_order = self.model_specification.parameter_names

        independent_variables = np.empty(
            (number_of_independents, number_of_model_parameters + 1), dtype=np.float32
        )

        dependent_variables = np.empty((number_of_independents), dtype=np.float32)

        dependent_variable_errors = np.empty(
            (number_of_independents, 2), dtype=np.float32
        )

        self.ordering = []
        filled_lines = 0

        for unique_identifier in unique_identifiers:
            self.ordering.append(unique_identifier)

            # Unpack model parameters into an array
            model_parameter_array = np.array(
                [
                    model_parameters[unique_identifier][parameter]
                    for parameter in parameter_order
                ]
            )

            this_model = model_values[unique_identifier]
            model_independent = this_model["independent"]
            model_dependent = this_model["dependent"]
            model_error = this_model.get(
                "dependent_error", np.zeros((len(model_independent), 2))
            )

            if np.ndim(model_error) == 1:
                # Need to force this into equal errors up and down.
                new_model_error = np.empty(
                    (len(model_independent), 2), dtype=np.float32
                )
                new_model_error[:, 0] = model_error[:]
                new_model_error[:, 1] = model_error[:]

                model_error = new_model_error

            for line in range(len(model_independent)):
                independent_variables[filled_lines][0] = model_independent[line]
                independent_variables[filled_lines][1:] = model_parameter_array

                dependent_variables[filled_lines] = model_dependent[line]
                dependent_variable_errors[filled_lines] = model_error[line]

                filled_lines += 1

        assert filled_lines == number_of_independents

        self.independent_variables = independent_variables
        self.dependent_variables = dependent_variables
        self.dependent_variable_errors = dependent_variable_errors


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

        return GaussianProcessEmulator(
            model_specification=self.model_specification,
            model_parameters=self.model_parameters,
            model_values=model_values,
        )

