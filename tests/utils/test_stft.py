import numpy as np
import pytest
from numpy.typing import NDArray

from adaptive_filter.utils.stft import STFT


def test_stft():
    N = 1024
    x = np.random.randn(5000)
    stft = STFT(window_length=N, window_type="hann")

    x_f = stft.stft(x)
    assert isinstance(x_f, np.ndarray)
    # x_hat = stft.istft(x_f)
    #
    # x_hat = x_hat[: len(x_f)]
    #
    # err = x - x_hat
    # assert err == 0
