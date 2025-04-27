import numpy as np
import pytest

from adaptive_filter.algorithms.fd_nlms import FDNLMS


# creating an expected output to test output
@pytest.mark.parametrize(
    "mu, x_f, e_f, expected_result",
    [
        (0.5, np.array([2.0 + 0j]), np.array([5.0 + 0j]), np.array([1.25 + 0j])),
        (1.0, np.array([1.0 + 1j]), np.array([2.0 - 1j]), np.array([(1.0 - 3j) / 2])),
    ],
)
def test_update_step(mu, x_f, e_f, expected_result):
    # creating filter object
    filter = FDNLMS(mu=mu, n=len(x_f), block_size=32)
    output = filter.update_step(e_f, x_f)
    assert isinstance(output, np.ndarray)
    assert output.shape == expected_result.shape
    np.testing.assert_allclose(output, expected_result, rtol=1e-5, atol=1e-6)
