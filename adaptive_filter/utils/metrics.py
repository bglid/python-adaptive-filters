"""
EVALUATION METRICS: MSE, SNR, Convergence (rate and time), Misadjustment, Clock time (for algorithm complexity)
"""

from typing import Any

import collections
import time

import numpy as np

from adaptive_filter.utils.math_utils import EigenDecomposition


class EvaluationSuite:
    def __init__(self, algorithm: str) -> None:
        self.algorithm = algorithm

    def MSE(self, desired_signal: float, input_signal: float) -> np.float64:
        """Calculates the Mean Squared Error = 1/n * sum(y - y_hat)**2

        Args:
            desired_signal (float): Observed or desired value
            input_signal (float): Prediction of value

        Returns:
            np.float64: Mean squared error
        """
        return np.mean((desired_signal - input_signal) ** 2)

    def SNR(self, desired_signal: float, noisy_signal: float) -> Any:
        """Calculates the Signal to Noise Ratio in dB: SNR = (Power of Signal)/(Power of Noise).

        Args:
            desired_signal (float): Input Signal (usually clean)
            processed_signal (float): Input Noisy speech (error usually)

        Returns:
            Any: SNR in dB
        """
        # formula is: snr = 10log_10((s)^2 / (s - s_hat)^2)
        "$$SNR = 10 log_{10}\\frac{s^{⊤}s}{(s − ˆs)^{⊤}(s − ˆs)}$$"
        # Need to account for when speech is silent...

        energy_thresh = 1e-6
        signal_power = np.mean(desired_signal**2)
        # returning zero if signal power is less than the energy thresh, meaning no speech
        if signal_power < energy_thresh:
            return 0
        # calculating the residual noise: clean - error output
        noise_power = np.mean((desired_signal - noisy_signal) ** 2) + 1e-12
        # Now we can return the SNR ratio in dB
        snr = signal_power / (noise_power)
        # print(f"Global SNR: {10 * np.log10(snr)}")
        return 10 * np.log10(snr + 1e-12)

    # function for Convergance
    def lms_convergence_rate(
        self,
        step_size: float,
        input_signal: np.ndarray[Any, np.dtype[Any]],
        k: int,
    ) -> Any:
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
    def evaluation(
        self,
        desired_signal,
        input_signal,
        step_size,
        time_k,
        error_output=None,
        clean_signal=None,
    ):
        """Calculates and returns suite of evaluation metrics

        Args:
            desired_signal (float): Input Signal
            input_signal (float): Input Noise
            step_size (float): Step size Mu
            time_k (int): the point in time in samples
            error_output (float): Error output of d - y, ideally cleaner signal
            clean_signal (float): Clean Signal

        Returns:
            dict: Dictionary of evaluation results
        """

        eval_results: dict[str, list[Any]] = {
            "MSE": [],
            "SNR": [],
        }

        eval_results["MSE"].append(self.MSE(desired_signal, input_signal))
        # LMS eval
        if self.algorithm == "LMS":
            # For SNR, desired signal is actually the clean signal and noisy signal is the error output
            eval_results["SNR"].append(
                self.SNR(desired_signal=clean_signal, noisy_signal=error_output)
            )
            # eval_results["Convergence"] = self.lms_convergence_rate(
            #     step_size, input_signal, k=time_k
            # )

        return eval_results


if __name__ == "__main__":

    evaluation = EvaluationSuite(algorithm="LMS")
