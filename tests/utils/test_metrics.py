import numpy as np
import pytest

from adaptive_filter.utils.metrics import EvaluationSuite


def test_SNR():
    d = np.array([1.0, 2.0])
    # for thresh test
    d_low = np.array([1e-10, 1e-10])
    noisy_sig = np.array([-1.0, -1.0])

    snr = EvaluationSuite("LMS")
    assert snr.algorithm == "LMS"
    output = snr.SNR(d_low, noisy_sig)
    # testing threshold works
    assert output == 0
