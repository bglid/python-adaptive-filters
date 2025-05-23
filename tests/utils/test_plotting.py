from types import ModuleType

import matplotlib
import numpy as np
import pytest

from adaptive_filter.utils.plotting import PlotSuite

matplotlib.use("Agg")


# creating a sample pytest model
@pytest.fixture
def sample_model():
    # creating a sample model
    filter_plot = PlotSuite(algorithm="LMS", num_samples=10)
    return filter_plot


def test_signal_plot(sample_model):
    signal = np.arange(0.0, 1.0, 0.1)
    desc = "testing"

    # testing that we are returned none when subplot is false
    test_none = sample_model.signal_plot(signal=signal, description=desc, subplot=False)
    assert test_none is None

    # testing that we receive a plot otherwise
    test_subplot = sample_model.signal_plot(
        signal=signal, description=desc, subplot=True
    )
    assert isinstance(test_subplot, ModuleType)


def test_error_plot(sample_model):
    signal = np.arange(0.0, 1.0, 0.1)
    desc = "testing"

    # testing that we are returned none when subplot is false
    test_none = sample_model.error_plot(
        results=signal, error_metric=desc, subplot=False
    )
    assert test_none is None

    # testing that we receive a plot otherwise
    test_subplot = sample_model.error_plot(
        results=signal, error_metric=desc, subplot=True
    )
    assert isinstance(test_subplot, ModuleType)


def test_full_plot_suite(sample_model):
    signal = np.arange(0.0, 1.0, 0.1)
    desc = "testing"
    result = np.arange(0.0, 1.0, 0.3)
    error_metric = "error_metric_test"

    # testing that we receive a plot otherwise
    test_plots = sample_model.full_plot_suite(
        signal=signal, description=desc, results=result, error_metric=error_metric
    )
    assert test_plots is None
