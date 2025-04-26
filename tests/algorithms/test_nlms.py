import numpy as np
import pytest

from adaptive_filter.algorithms import nlms
from adaptive_filter.filter_models.filter_model import FilterModel


# creating an expected output to test output
@pytest.mark.parametrize(
    "mu,e_n,x_n, expected_result",
    [
        (0.5, 2.0, np.array([1.0, -1.0]), np.array([0.5, -0.5])),
        (1.0, 1.0, np.array([5.0, 2.0]), np.array([0.172414, 0.068966])),
        (-1.0, 1.0, np.array([0.5, 0.25]), np.array([-1.6, -0.8])),
    ],
)
def test_update_step(mu, e_n, x_n, expected_result):
    # creating filter object
    filter = nlms.NLMS(mu=mu, n=len(x_n))
    output = filter.update_step(e_n, x_n)
    assert isinstance(output, np.ndarray)
    np.testing.assert_allclose(output, expected_result, rtol=1e-5, atol=1e-6)
