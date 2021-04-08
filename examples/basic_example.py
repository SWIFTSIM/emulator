"""
A quick example of how to use the swift emulator to 'predict'
values in a non-linear model.
"""

import swiftemulator as se
import numpy as np

np.random.seed(5330)

import sys

try:
    test_model = float(sys.argv[1])

    if test_model > 1.0 or test_model < 0.0:
        raise AttributeError
except:
    print("Please include a first parameter between 0.0 and 1.0")
    exit(0)


def my_true_model(x, offset):
    """
    A basic non-linear model.
    """
    return offset + offset ** 2 * x ** 0.5


model_specification = se.ModelSpecification(
    number_of_parameters=1,
    parameter_names=["m"],
    parameter_limits=[[0.0, 1.0]],
    parameter_printable_names=["Offset"],
)

# Have ten independent models
number_of_models = 10

model_parameters = se.ModelParameters(
    model_parameters={
        run_number: {"m": gradient}
        for run_number, gradient in enumerate(np.random.rand(10))
    }
)

# Simulate outputs from the ten independent models
number_of_model_samples = 8

model_values = {}

for run_number in range(number_of_models):
    independent = np.random.rand(number_of_model_samples)
    # Dependent includes some scatter
    dependent = my_true_model(
        independent, model_parameters.model_parameters[run_number]["m"]
    ) + (np.random.rand(number_of_model_samples) * 0.2 - 0.1)
    # Constant errors
    dependent_errors = np.ones(number_of_model_samples, dtype=np.float32) * 0.05

    model_values[run_number] = {
        "independent": independent,
        "dependent": dependent,
        "dependent_error": dependent_errors,
    }

model_values = se.ModelValues(model_values=model_values)


# Now perform the regression with different emulators.

gpe_no_linear = se.emulators.gaussian_process.GaussianProcessEmulator()
gpe_no_linear.fit_model(
    model_specification=model_specification,
    model_parameters=model_parameters,
    model_values=model_values,
)

gpe_with_linear = se.emulators.gaussian_process.GaussianProcessEmulator(
    mean_model=se.mean_models.LinearMeanModel()
)
gpe_with_linear.fit_model(
    model_specification=model_specification,
    model_parameters=model_parameters,
    model_values=model_values,
)

lass = se.emulators.linear_model.LinearModelEmulator(lasso_model_alpha=0.0)
lass.fit_model(
    model_specification=model_specification,
    model_parameters=model_parameters,
    model_values=model_values,
)

example_independent = np.sort(np.random.rand(100))

y, yerr = gpe_no_linear.predict_values(
    example_independent, model_parameters={"m": test_model}
)
yerr *= 100

y_l, yerr_l = gpe_with_linear.predict_values(
    example_independent, model_parameters={"m": test_model}
)
yerr_l *= 100

y_l_only, _ = lass.predict_values(
    example_independent, model_parameters={"m": test_model}
)


import matplotlib.pyplot as plt

plt.figure(figsize=(4, 4), constrained_layout=True)

plt.fill_between(example_independent, y + yerr, y - yerr, alpha=0.5, color="C0")
plt.fill_between(example_independent, y_l + yerr_l, y_l - yerr_l, alpha=0.5, color="C1")


plt.plot(example_independent, y, label="Predicted (No Linear)", color="C0")
plt.plot(example_independent, y_l, label="Predicted (With Linear)", color="C1")
plt.plot(
    example_independent,
    y_l_only,
    label="Predicted (Only LM)",
    color="C2",
)

plt.plot(
    example_independent,
    my_true_model(example_independent, test_model),
    color="k",
    linestyle="dashed",
    label="True",
)

plt.ylabel(f"Predicted $y = {test_model:3.3f} + {test_model**2:3.3f} \\sqrt{{x}}$")
plt.xlabel("$x$")

plt.legend(loc="lower right")

try:
    filename = sys.argv[2]
    plt.savefig(filename)
except:
    print("You can save your result by providing the second parameter as the filename.")
    plt.show()
