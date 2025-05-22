"""
EVALUATION METRICS: MSE, SNR, Convergence, Clock time (for algorithm complexity)
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray


class EvaluationSuite:
    def __init__(self, algorithm: str) -> None:
        self.algorithm = algorithm

    def signal_reshaper(
        self,
        desired_signal: NDArray[np.float64],
        input_signal: NDArray[np.float64],
        clean_signal: NDArray[np.float64],
        error_signal: NDArray[np.float64],
    ) -> tuple:
        """Flattens/ravels the signals and truncates them to the shortest signal for downstream metric evals.

        Args:
            desired_signal (NDArray[np.float64]): Observed or desired value
            input_signal (NDArray[np.float64]): Prediction of value
            clean_signal (NDArray[np.float64]): Clean signal
            error_signal (NDArray[np.float64]):
                Error output of filter prediction (d - y)

        Returns:
            tuple:
                - NDArray[np.float64]: Shortened 1D desired signal
                - NDArray[np.float64]: Shortened 1D input y_n signal
                - NDArray[np.float64]: Shortened 1D clean signal
                - NDArray[np.float64]: Shortened 1D error signal
        """
        # first raveling each input
        d_flat = desired_signal.ravel()
        y_flat = input_signal.ravel()
        clean_flat = clean_signal.ravel()
        error_flat = error_signal.ravel()

        # finding the shortest signal
        shortest = min(
            len(d_flat),
            len(y_flat),
            len(clean_flat),
            len(error_flat),
        )

        # trimming each signal to the shortest length
        return (
            d_flat[:shortest],
            y_flat[:shortest],
            clean_flat[:shortest],
            error_flat[:shortest],
        )

    def MSE(
        self, desired_signal: NDArray[np.float64], input_signal: NDArray[np.float64]
    ) -> np.float64:
        """Calculates the Mean Squared Error = 1/n * sum(y - y_hat)**2

        Args:
            desired_signal (NDArray[np.float64]): Observed or desired value
            input_signal (NDArray[np.float64]): Prediction of value

        Returns:
            np.float64: Mean squared error
        """
        return np.mean((desired_signal - input_signal) ** 2)

    def SNR(
        self, desired_signal: NDArray[np.float64], noisy_signal: NDArray[np.float64]
    ) -> Any:
        """Calculates the Signal to Noise Ratio in dB: SNR = (Power of Signal)/(Power of Noise).

        Args:
            desired_signal (float): Input Signal (usually clean)
            noisy_signal(float): Input Noisy speech (error usually)

        Returns:
            Any: SNR in dB
        """
        # Need to account for when speech is silent...
        energy_thresh = 1e-6
        signal_power = np.sum(desired_signal**2)
        # returning zero if signal power is less than the energy thresh, meaning no speech
        if signal_power < energy_thresh:
            return 0
        # calculating the residual noise: clean - error output
        noise_power = np.sum((desired_signal - noisy_signal) ** 2) + 1e-12
        # Now we can return the SNR ratio in dB
        snr = signal_power / noise_power
        # print(f"Global SNR: {10 * np.log10(snr)}")
        return 10 * np.log10(snr + 1e-12)

    # function for Convergance
    def convergence_time(
        self,
        error: NDArray[np.float64],
        fs: int,
        samples_steady: int,
        r_tol: float,
        consecutive_samples: int,
    ) -> np.float64:
        """Time in seconds at it takes for the error to first fall into the steady-state convergence.

        Args:
            error (NDArray[np.float64]): Signal Error
            fs (int): Sampling rate
            samples_steady (int): number of last samples to average for steady-state.
            r_tol (float): Relative tolerance
            consecutive_samples (int): Number of consec. samples needed to delcare convergence.

        Returns:
            np.float64: Convergence time in seconds.
        """

        # get the steady error
        steady_error = np.mean(np.abs(error[-samples_steady:]))

        # set the error threshold
        thresh = (1 + r_tol) * steady_error
        below_thresh = np.abs(error) <= thresh

        # checking for different lengths of consec samples
        if consecutive_samples > 1:
            #  compute rowwing sum of below over length of consec samples
            summation = np.cumsum(np.concatenate([[0], below_thresh]))
            for i in range(len(error) - consecutive_samples + 1):
                if (
                    summation[i + consecutive_samples] - summation[i]
                ) == consecutive_samples:
                    return i / fs  # returning in seconds
            # else return nan
            return np.nan

        # else when consec is just 1
        else:
            id = np.where(below_thresh)[0]
            if id.size == 0:
                return np.nan
            return id[0] / fs
