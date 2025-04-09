"""
EVALUATION METRICS: MSE, SNR, Convergence (rate and time), Misadjustment, Clock time (for algorithm complexity)
"""

import collections
import time

import numpy as np

from adaptive_filter.utils.math_utils import EigenDecomposition


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
    def lms_convergence_rate(
        self,
        step_size,
        input_signal,
        k,
    ):
        """Calculates the first-order convergance rate of the mean-square error dependent on mu and autocorrelation eigenvalue.
        This process is done by first taking the autocorrelation (R) of the input data, X^K, where K indicates time.
        Then, eigendecomposition is run to get the the min eigenvalue, indicating the slowet possible decay,
        which is used in the final calc. (1 - 2 * step_size * lambda_i)^K
        """
        # first we get the autocorrelation matrix R = X^{k}
        autocorrelation = np.dot(input_signal, input_signal.T)
        # Running Eigendecomposition on the auto correlation
        eigen = EigenDecomposition(k_iterations=25)
        W, lambda_matrix, eigenvalues = eigen.eigendecomposition(
            covariance_matrix=autocorrelation, n_eigenvectors=10
        )
        # getting the smallest eigenvalue for our convergence rate after K time
        max_lambda = np.min(eigenvalues)
        return np.abs(1 - 2 * step_size * max_lambda) ** k

    # function to run full evaulation/benchmark suite
    def evaluation(self, desired_signal, input_signal, step_size, time_k):
        """Calculates and returns suite of evaluation metrics

        Args:
            desired_signal (float): Input Signal
            input_signal (float): Input Noise

        Returns:
            dict: Dictionary of evaluation results
        """

        eval_results = collections.defaultdict(float)

        eval_results["MSE"] = self.MSE(desired_signal, input_signal)


if __name__ == "__main__":

    evaluation = EvaluationSuite
