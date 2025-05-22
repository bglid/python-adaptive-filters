import numpy as np
import pytest

from adaptive_filter.utils.realNoiseSimulator import mic_white_noise, reference_delay


@pytest.mark.parametrize("snr_db", [0.1, 10.0, 50.0])
def test_mic_white_noise(snr_db):
    N = 1024
    # noise_ref = np.full(N, 0.5, dtype=np.float64)
    noise_ref = np.full(100000, 2.0, dtype=np.float64)
    ref_power = np.mean(noise_ref**2)
    output = mic_white_noise(noise_ref, snr_input=snr_db)

    assert output.shape == noise_ref.shape
    assert output.dtype == np.float64

    noise_added = output - noise_ref
    added_power = np.mean(noise_added**2)

    estimated_snr = 10 * np.log10(ref_power / added_power)

    assert pytest.approx(snr_db, rel=0.1, abs=0.2) == estimated_snr


def test_reference_delay():
    # testing that 0 noise delay doesn't delay signal
    noise_ref = np.array([0.1, 0.2, 0.3])
    delay_amount = 0
    fs = 44100
    non_delayed_sig = reference_delay(noise_ref, delay_amount, fs)
    np.testing.assert_almost_equal(non_delayed_sig, noise_ref)
