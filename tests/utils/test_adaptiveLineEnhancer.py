import numpy as np
import pytest

from adaptive_filter.utils.adaptiveLineEnhancer import adaptive_line_enhancer


@pytest.mark.parametrize(
    "x, D, expected", [(np.arange(5), 2, np.array([0.0, 0.0, 0.0, 1.0, 2.0]))]
)
def test_ale(x, D, expected):
    output_result = adaptive_line_enhancer(d=x, ale_delay=D)
    assert output_result.shape == x.shape
    # assert output_result.shape == expected
    np.testing.assert_array_equal(output_result, expected)
