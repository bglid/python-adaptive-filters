"""
EVALUATION METRICS: MSE, SNR, Convergence (rate and time), Misadjustment, Clock time (for algorithm complexity)
"""

import time

import numpy as np


class EvaluationSuite:
    def __init__(self, algorithm: str) -> None:
        self.algorithm = algorithm

    # function for MSE
    def MSE(self, y: float, y_hat: float):
        """Calculates the Mean Squared Error = 1/n * sum(y - y_hat)**2

        Args:
            y (float): Observed or desired value
            y_hat (float): Prediction of value

        Returns:
            float: Mean squared error
        """
        return np.mean(y - y_hat) ** 2

    # function for SNR
    def SNR(self, signal: float, noise: float):
        """Calculates the Signal to Noise Ratio in dB: SNR = (Power of Signal)/(Power of Noise).

        Args:
            signal (float): Input Signal
            noise (float): Input Noise

        Returns:
            float: SNR Ratio in dB
        """

        # First we need to measure signal power and noise power
        signal_power = np.mean(signal**2)
        noise_power = np.mean(noise**2)
        # Now we can return the SNR ratio in dB
        snr = signal_power / noise_power
        return 10 * np.log10(snr)

    # function for Convergance
    def convergence_rate(
        self,
        step_size,
    ):
        """Calculates the convergance rate dependent on mu and autocorrelation eigenvalue"""
        pass

    # function to run full evaulation/benchmark suite
    def evaluation(self, desired_signal, output_signal):
        pass


if __name__ == "__main__":

    evaluation = EvaluationSuite
