import numpy as np
import pytest

from adaptive_filter.algorithms import apa


# creating an expected output to test output
@pytest.mark.parametrize(
    "mu,error_vec,x_mat, expected_result",
    [
        (0.5, np.array([2.0, 3.0]), np.eye(2), np.array([1.0, 1.5])),
        (1.0, np.array([5.0, 2.0]), np.eye(2), np.array([5.0, 2.0])),
        (
            0.1,
            np.array([1.0, 2.0]),
            np.array([[2.0, 0.0], [0.0, 3.0]]),
            0.1 * np.array([0.5, 0.6666667]),
        ),
    ],
)
def test_update_step(mu, error_vec, x_mat, expected_result):
    # need to get columns in x_mat
    # creating filter object
    ap_filter = apa.APA(mu=mu, n=x_mat.shape[0])
    output = ap_filter.update_step(error_vec, x_mat)
    assert isinstance(output, np.ndarray)
    assert output.shape == (x_mat[0].shape)
    np.testing.assert_allclose(output, expected_result, rtol=1e-6, atol=1e-8)
