"""
Tests the models in mean_models.
"""

from swiftemulator.mean_models.offset import OffsetMeanModel
from swiftemulator.mean_models.linear import LinearMeanModel
from swiftemulator.mean_models.polynomial import PolynomialMeanModel

import numpy as np


def test_offset_mean_model():
    mean_model = OffsetMeanModel()

    mean_model.train(independent=np.random.rand(100, 23), dependent=np.arange(100))

    predicted = mean_model.predict(
        independent=np.random.rand(12, 23),
    )

    assert (predicted == 49.5).all()


def test_linear_mean_model():
    mean_model = LinearMeanModel(lasso_model_alpha=1.0)

    mean_model.train(independent=np.random.rand(100, 10), dependent=np.arange(100))

    mean_model.predict(
        independent=np.random.rand(12, 10),
    )

    george_model = mean_model.george_model

    george_model.get_value(np.random.rand(12, 10))


def test_polynomial_mean_model():
    mean_model = PolynomialMeanModel(degree=4)

    mean_model.train(independent=np.random.rand(100, 10), dependent=np.arange(100))

    mean_model.predict(
        independent=np.random.rand(12, 10),
    )

    george_model = mean_model.george_model

    george_model.get_value(np.random.rand(12, 10))
