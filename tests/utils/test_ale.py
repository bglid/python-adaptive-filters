import numpy as np
import pytest

from adaptive_filter.utils.ale import ale


@pytest.mark.parametrize(
    "x, D, expected", [(np.arange(5), 2, np.array([0.0, 0.0, 0.0, 1.0, 2.0]))]
)
def test_ale(x, D, expected):
    output_result = ale(d=x, ale_delay=D)
    assert output_result.shape == x.shape
    # assert output_result.shape == expected
    np.testing.assert_array_equal(output_result, expected)
