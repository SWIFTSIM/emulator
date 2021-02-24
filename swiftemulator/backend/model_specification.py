"""
Model specification for the emulator.

Contains the :class:``ModelSpecification`` class,
used to input the parameter names and limits.
"""

import attr
from typing import List, Optional


@attr.s
class ModelSpecification(object):
    """
    Base specification for the model. Contains information about
    the names of parameters and their ranges.

    Parameters
    ----------

    number_of_parameters, int
        Total number of variable parameters in your model.

    parameter_names, List[str]
        The names of parameters in your model; these will be
        used to access the parameters in :class:``ModelParameters``.

    parameter_limits, List[List[float]]
        The lower and upper limit of the input parameters. Should be
        the same length as ``parameter_names``, but each item is a list
        of length two, with a lower and upper bound. For example, in
        a two parameter model ``[[0.0, 1.0], [8.3, 9.3]]`` would mean
        that the first parameter would vary between 0.0 and 1.0, with
        the second parameter varying between 8.3 and 9.3.

    parameter_printable_names, List[str], optional
        Optional 'fancy' names for your parameters. These strings will
        be used on any figures generated through swift-emulator. Can
        include LaTeX formatting as in ``matplotlib``.

    Raises
    ------

    AttributeError
        When the number of paremeters in all of the required attributes
        are not equal (e.g. a different number of names has been provided
        compared to the number of limits).

    """

    number_of_parameters: int = attr.ib(validator=attr.validators.instance_of(int))
    parameter_names: List[str] = attr.ib()
    parameter_limits: List[List[float]] = attr.ib()
    parameter_printable_names: Optional[List[str]] = attr.ib(default=None)

    @parameter_limits.validator
    def _check_parameter_limits(self, attribute, value):
        if not len(self.parameter_names) == len(value):
            raise AttributeError(
                "Parameter limits and parameter names not of same length."
            )

    def __attrs_post_init__(self):
        if not self.number_of_parameters == len(self.parameter_names):
            raise AttributeError(
                "Number of parameters does not match the length of parameter names"
            )
        if self.parameter_printable_names is None:
            self.parameter_printable_names = self.parameter_names

    @property
    def salib_problem(self):
        """
        Generates the ``SALib`` ``problem`` dictionary.
        """

        problem = {
            "num_vars": self.number_of_parameters,
            "names": self.parameter_names,
            "bounds": self.parameter_limits,
        }

        return problem
