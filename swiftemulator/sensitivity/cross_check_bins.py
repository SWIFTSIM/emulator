"""
Basic test for any emulator. Leave one simulation out
and test how well the emulator fits those values.
"""

import attr
import numpy as np
import matplotlib.pyplot as plt
import george
import copy
import sklearn.linear_model as lm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from pathlib import Path

from swiftemulator.backend.model_parameters import ModelParameters
from swiftemulator.backend.model_values import ModelValues
from swiftemulator.backend.model_specification import ModelSpecification

from swiftemulator.emulators.gaussian_process_bins import GaussianProcessEmulatorBins


from typing import List, Dict, Tuple, Union, Optional, Hashable


@attr.s
class CrossCheckBins(object):
    """
    Generator for emulators for leave one out checks.

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

    leave_out_order: Optional[List[int]] = None
    cross_emulators: Optional[Dict[Hashable, george.GP]] = None

    def build_emulators(
        self,
        kernel=None,
        fit_model: str = "none",
        lasso_model_alpha: float = 0.0,
        polynomial_degree: int = 1,
        hide_progress: bool = True,
    ):
        """
        Build a dictonary with an emulator for each simulation
        where the data of that simulation is left out

        Note: this can take a long time

        Parameters
        ----------

        kernel, george.kernels
            The ``george`` kernel to use. The GPE here uses a copy
            of this instance. By default, this is the
            ``ExpSquaredKernel`` in George

        fit_model, str
            Type of model to use for mean fitting, Optional, defaults
            to none which is a pure GP modelling. Options: "linear" and
            "polynomial"

        lasso_model_alpha, float
            Alpha for the Lasso model (only used of course when asking to
            ``fit_linear_model``). If this is 0.0 (the default) basic linear
            regression is used.

        polynomial_degree, int
            Maximal degree of the polynomial surface, default 1; linear for each
            parameter

        hide_progress: bool
            Option to display a tqdm bar when creating the emulators,
            Default is hide progress bar
        """

        model_values = self.model_values
        leave_out_order = list(model_values.model_values.keys())
        self.leave_out_order = leave_out_order

        emulators = {}

        for unique_identifier in tqdm(self.leave_out_order, disable=hide_progress):
            left_out_data = model_values.model_values.pop(unique_identifier)

            emulator = GaussianProcessEmulatorBins(
                model_specification=self.model_specification,
                model_parameters=self.model_parameters,
                model_values=model_values,
            )

            emulator.build_arrays()
            emulator.fit_model(
                kernel=kernel,
                fit_model=fit_model,
                lasso_model_alpha=lasso_model_alpha,
                polynomial_degree=polynomial_degree,
            )

            emulators[unique_identifier] = emulator

            model_values.model_values[unique_identifier] = left_out_data

        self.cross_emulators = emulators

        return

    def plot_results(self, output_path: Optional[Union[str, Path]] = None):
        """
        Make a plot of each of the leave_out emulators vs
        the original data.

        Parameters
        ----------

        emulate_at: np.array
            independent array where the emulator is evaluated.

        foldername: str, None
            Optional, name of the folder where you want to save
            the figures.
        """
        for unique_identifier in self.cross_emulators.keys():
            fig, ax = plt.subplots()

            emulate_at, emulated, emulated_error = self.cross_emulators[
                unique_identifier
            ].predict_values(
                model_parameters=self.model_parameters.model_parameters[
                    unique_identifier
                ],
            )

            ax.fill_between(
                emulate_at,
                emulated - np.sqrt(emulated_error),
                emulated + np.sqrt(emulated_error),
                color="C1",
                alpha=0.3,
                linewidth=0.0,
            )

            ax.errorbar(
                self.model_values.model_values[unique_identifier]["independent"],
                self.model_values.model_values[unique_identifier]["dependent"],
                yerr=self.model_values.model_values[unique_identifier][
                    "dependent_error"
                ],
                label="True",
                marker=".",
                linestyle="none",
                color="C0",
            )

            ax.plot(emulate_at, emulated, label="Emulated", color="C1")

            plt.xlabel("Log($M_*$ / M$_\odot$)")
            plt.ylabel("Log(Mass Function)")
            plt.legend()
            plt.title(f"Leave Out Run {unique_identifier}")
            if foldername is None:
                plt.show()
            else:
                plt.savefig(Path(output_path) / f"leave_out_{unique_identifier}.png")

    def get_mean_squared(
        self,
        use_dependent_error: bool = False,
        use_y_as_error: bool = False,
        use_squared_difference: bool = True,
    ):
        """
        Calculates the mean squared per simulation and the total mean squared
        of the entire set of left-out simulations.

        Parameters
        ----------

        use_dependent_error: bool
            Use the simulation errors as weights for the mean squared calulation.
            Default is false.

        use_y_as_error: boolean
            Use the model y values as the weights for the calculation.

        use_squared_difference: boolean
            Use the simulation errors as weights for the mean squared calulation.
            Default is false.
        """
        mean_squared_dict = {}
        total_mean_squared = np.array([])
        for unique_identifier in self.cross_emulators.keys():
            x_model = self.model_values.model_values[unique_identifier]["independent"]
            y_model = self.model_values.model_values[unique_identifier]["dependent"]

            _, emulated, _ = self.cross_emulators[unique_identifier].predict_values(
                model_parameters=self.model_parameters.model_parameters[
                    unique_identifier
                ],
            )

            emulated = emulated[: len(x_model)]

            if use_y_as_error:
                y_model_error = y_model
            else:
                y_model_error = self.model_values.model_values[unique_identifier][
                    "dependent_error"
                ]

            if use_dependent_error:
                uniq_mean_squared = (y_model - emulated) / y_model_error
            else:
                uniq_mean_squared = y_model - emulated

            if use_squared_difference:
                uniq_mean_squared = uniq_mean_squared ** 2

            mean_squared_dict[unique_identifier] = uniq_mean_squared
            total_mean_squared = np.append(total_mean_squared, uniq_mean_squared)

        return np.mean(total_mean_squared), mean_squared_dict
