# Class that contains filter model used by most adaptive filters
from typing import Any

import time

import numpy as np
from numpy.typing import NDArray

from adaptive_filter.utils.metrics import EvaluationSuite


class FilterModel:
    def __init__(self, mu: float, filter_order: int) -> None:
        # consider adding p: order
        self.mu = mu  # step_rate
        self.N = filter_order  # filter window size
        # Algorithm type, defined by subclass algorithm
        self.algorithm = ""

    def noise_estimate(self, x_n: NDArray[np.float64]) -> np.float64:
        """Predicts the noise estimate, given vector X[n], noise reference. Uses formula W^T[n]X[n]

        Args:
            x_n (NDArray[np.float64]): vector[n] of array X, the noise estimate

        Returns:
            np.float64: Predicted noise estimate output of the FIR filter and the noise reference
        """
        return np.dot(self.W, x_n)

    def error(self, d_n: float, noise_estimate: float) -> float:
        """Calculates the error, e[n] = d[n] - y[n], y[n] is output of W^T[n]X[n]

        Args:
            d_n (float): Desired sample at point n of array D, noisy input
            noise_estimate (float): The noise estimate product (y[n])

        Returns:
            float: error of (noisy) desired input[n] - noise estimate. Ideally, this should be the clean signal
        """
        return d_n - noise_estimate

    def update_step(self, e_n: float, x_n: NDArray[np.float64]) -> NDArray[np.float64]:
        """Updates weights of W[n + 1], given the learning algorithm chosen

        Args:
            e_n (float): Error sample at point n
            x_n (NDArray[np.float64]): Input vector n

        Returns:
            NDArray[np.float64]: Update step to self.W
        """
        return np.zeros(len(x_n))

    def filter(
        self,
        d: NDArray[np.float64],
        x: NDArray[np.float64],
        clean_signal: NDArray[np.float64],
    ) -> tuple:
        """Iterates Adaptive filter alorithm and updates for length of input signal X

        Args:
            d (NDArray[np.float64]):
                "Desired Signal", which in the ANC use-case is the noisy input signal.
            x (NDArray[np.float64]):
                Input reference matrix X, which in the ANC case is the noise reference.
            clean_signal (NDArray[np.float64]):
                Clean signal for final reference.

        Returns:
            tuple: A tuple containing:
                - NDArray[np.float64]: "Clean output" The error signal of d - y.
                - NDArray[np.float64]: Predicted noise estimate.
                - float: Mean of the Adaption MSE across the signal.
                - float: Mean of the Speech MSE across the signal.
                - float: Global SNR across the signal.
                - float: Delta SNR improvement
                - float: Clock-time of filter performance
        """

        # initializing our weights given X
        self.W = np.random.normal(0.0, 0.5, x[0].shape)
        self.W *= 0.001  # setting weights close to zero
        if self.W.ndim <= 1:
            self.W = self.W.reshape(-1, 1)
            # print(self.W.shape)
        assert self.W.ndim == 2

        # turning D and X into np arrays, if not already
        if type(d) is not np.ndarray:
            d = np.array(d)
        # asserting the shape of d
        if d.ndim == 1:
            d = d.reshape(-1, 1)  # making shape (n, 1)
        assert d.ndim == 2

        if type(x) is not np.ndarray:
            x = np.array(x)
        # checking X shape
        if x.ndim == 1:
            x = x.reshape(-1, 1)  # making shape (n, 1)
        assert x.ndim == 2

        # need to truncate noise estimate if it's longer than the desired singal
        if d.shape[0] < x.shape[0]:
            x = x[: d.shape[0]]
        if clean_signal.shape[0] < d.shape[0]:
            d = d[: clean_signal.shape[0]]
            x = x[: clean_signal.shape[0]]

        # getting the number of samples from x len
        num_samples = len(x)

        # creating evaluation object
        evaluation_runner = EvaluationSuite(algorithm=self.algorithm)

        # initializing the arrays to hold error and noise estimate
        noise_estimate = np.zeros(num_samples)
        error = np.zeros(num_samples)

        # creating an array to track the weight changes over time N
        # self.weight_t = np.zeros(())

        # clock-time for how long filtering this signal takes
        start_time = time.perf_counter()
        for sample in range(num_samples):
            # getting the prediction y (noise estimate)
            noise_estimate[sample] = self.noise_estimate(x[sample])
            # getting the error e[sample] = d[sample] - y[sample]
            error[sample] = self.error(
                d_n=d[sample], noise_estimate=noise_estimate[sample]
            )
            # updating the weights
            self.W += self.update_step(e_n=error[sample], x_n=x[sample])

        # taking clock-time before running metrics
        elapsed_time = time.perf_counter() - start_time

        # to avoid memory issues, need to ensure signals are same shape
        d_flat, y_flat, clean_flat, error_flat = evaluation_runner.signal_reshaper(
            d, noise_estimate, clean_signal, error
        )

        # What LMS minimized
        adaption_mse_result = evaluation_runner.MSE(d_flat, y_flat)
        # How close e[n] is to s[n]
        speech_mse_result = evaluation_runner.MSE(clean_flat, error_flat)
        # full SNR
        snr_result = evaluation_runner.SNR(clean_flat, error_flat)
        # # getting the Delta SNR
        snr_in = evaluation_runner.SNR(clean_flat, d_flat)  # SNR without filtering
        delta_snr = snr_result - snr_in

        return (
            error,
            noise_estimate,
            adaption_mse_result,
            speech_mse_result,
            snr_result,
            delta_snr,
            elapsed_time,
        )
