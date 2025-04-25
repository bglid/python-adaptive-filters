"""
Util to make the noise reference more imperfect.
Simluating the imperfections of real world noise referencing.
"""

import numpy as np
from numpy.typing import NDArray


# Adding real mic white noise
def mic_white_noise(
    noise_ref: NDArray[np.float64], snr_input: float = 30.0
) -> NDArray[np.float64]:
    """Attempts to further sully noise by simulating real mic pickup, adding more noise.

    Args:
        noise_ref (NDArray[np.float64]): vector[n] of array X, the noise estimate
        snr_input (float): The SNR ratio of the amount of noise to be added.

    Returns:
        NDArray[np.float64]: Noise reference with added noise.
    """
    # getting the power of the noise reference
    reference_power = np.mean(noise_ref**2)
    mic_noise_power = reference_power / (10 ** (snr_input / 10))

    # creating the mic "noise" in the form of SNR
    mic_noise = np.random.randn(noise_ref.shape[0]) * np.sqrt(mic_noise_power)

    # returning summation of noise and reference
    return noise_ref + mic_noise


# adding a slight delay to the signal
def reference_delay(
    noise_ref: NDArray[np.float64], delay_amount: float, fs: int
) -> NDArray[np.float64]:
    """Adds delay to simulate any mic input delay

    Args:
        noise_ref (NDArray[np.float64]): vector[n] of array X, the noise estimate
        delay_amount (float): Delay in ms.
        fs (int): Sampling rate.

    Returns:
        NDArray[np.float64]: Noise reference with added delay.
    """

    # converting the delay to seconds
    delay_s = delay_amount / 1000
    t = np.arange(len(noise_ref)) / fs
    # resample with a delay
    return np.interp(t, t - delay_s, noise_ref, left=0.0)
