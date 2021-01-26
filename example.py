"""
A quick example of how to use the swift emulator to 'predict'
values in a linear model.
"""

import swiftemulator as se
import numpy as np

import sys

try:
    test_model = float(sys.argv[1])

    if test_model > 1.0 or test_model < 0.0:
        raise AttributeError
except:
    print("Please include a first parameter between 0.0 and 1.0")
    exit(0)


def my_true_model(x, m):
    """
    A basic linear model.
    """
    return x * m


model_specification = se.ModelSpecification(
    number_of_parameters=1,
    parameter_names=["m"],
    parameter_limits=[[0.0, 1.0]],
    parameter_printable_names=["Gradient"],
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
number_of_model_samples = 24

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

# Now we can perform the regression.
generator = se.EmulatorGenerator(
    model_specification=model_specification, model_parameters=model_parameters
)

gpe = generator.create_gaussian_process_emulator(model_values=model_values)

gpe.build_arrays()

gpe.fit_model()

example_independent = np.sort(np.random.rand(100))

y, yerr = gpe.predict_values(example_independent, model_parameters={"m": test_model})
yerr *= 100

import matplotlib.pyplot as plt

plt.figure(figsize=(4, 4), constrained_layout=True)

plt.fill_between(example_independent, y + yerr, y - yerr, alpha=0.5)

plt.plot(example_independent, y, label="Predicted")
plt.plot(
    example_independent,
    my_true_model(example_independent, test_model),
    color="k",
    linestyle="dashed",
    label="True",
)

plt.ylabel(f"Predicted $y = {test_model} x$")
plt.xlabel("$x$")

plt.legend()

plt.show()
