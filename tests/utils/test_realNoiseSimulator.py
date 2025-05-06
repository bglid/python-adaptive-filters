import numpy as np
import pytest

from adaptive_filter.utils.realNoiseSimulator import mic_white_noise, reference_delay


def test_mic_white_noise():
    pass


def test_reference_delay():
    # testing that 0 noise delay doesn't delay signal
    noise_ref = np.array([0.1, 0.2, 0.3])
    delay_amount = 0
    fs = 44100
    non_delayed_sig = reference_delay(noise_ref, delay_amount, fs)
    np.testing.assert_almost_equal(non_delayed_sig, noise_ref)
