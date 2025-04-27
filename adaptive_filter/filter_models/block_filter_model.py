# Class that contains filter model used by most adaptive filters
from typing import Any

import time
from collections import deque

import numpy as np
from numpy.core.shape_base import block
from numpy.typing import NDArray

from adaptive_filter.utils.metrics import EvaluationSuite
from adaptive_filter.utils.stft import STFT


# SIMILAR to FilterModel, but with block-based processing logic added
class BlockFilterModel:
    def __init__(
        self,
        mu: float,
        filter_order: int,
        block_size: int,
    ) -> None:
        self.mu = mu
        self.N = filter_order
        # FFT length or P order for APA
        self.block_size = block_size
        self.hop_size = block_size // 2
        self.half_bins = self.block_size // 2 + 1
        self.eps = 1e-8
        # Algorithm type, defined by subclass algorithm
        self.algorithm = ""
        # # initializing weights
        # self.W = np.zeros(self.half_bins, dtype=np.complex128)

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

    def update_step(
        self,
        e_n: float,
        x_n: NDArray[np.float64],
    ) -> NDArray[np.float64]:
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

        Raises:
            ValueError: If Signal dims are not compatible (1D)
        """
        # initializing our weights given X
        self.W = np.random.normal(0.0, 0.5, self.N)
        self.W *= 0.001  # setting weights close to zero

        # turning D and X into np arrays, if not already
        d = np.asarray(d).ravel()
        if d.ndim != 1:
            raise ValueError(f"Expected desired signal to be 1D, got shape: {d.shape}")

        x = np.asarray(x).ravel()
        if x.ndim != 1:
            raise ValueError(f"Expected input signal to be 1D, got shape: {x.shape}")

        clean_signal = np.asarray(clean_signal).ravel()
        if clean_signal.ndim != 1:
            raise ValueError(
                f"Expected clean signal to be 1D, got shape: {clean_signal.shape}"
            )

        if d.shape[0] < x.shape[0]:
            x = x[: d.shape[0]]
            # assert x.shape == d.shape  # Double check
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

        # creating an array to track MSE history for convergence metrics
        mse_history = np.zeros(num_samples)

        # creating a ciruclar buffer for the filter taps
        circ_buffer = np.zeros(self.N, dtype=float)
        # for buffering blocks
        x_buffer = deque(maxlen=self.block_size)
        error_buffer = deque(maxlen=self.block_size)

        # clock-time for how long filtering this signal takes
        start_time = time.perf_counter()
        for sample in range(num_samples):
            # using a circular buffer style window technique:
            circ_buffer = np.roll(circ_buffer, 1)
            # writer-pointer to add the most recent sample into the N buffer window
            circ_buffer[0] = x[sample]

            # getting the prediction y (noise estimate)
            noise_estimate[sample] = self.noise_estimate(circ_buffer)
            # getting the error e[sample] = d[sample] - y[sample]
            error[sample] = self.error(
                d_n=d[sample], noise_estimate=noise_estimate[sample]
            )
            # Buffering blocks
            x_buffer.append(circ_buffer.copy())
            error_buffer.append(error[sample])

            # APA update,  hop = %50 block_size
            if len(x_buffer) == (self.block_size) and (sample % self.hop_size) == 0:
                if self.algorithm not in ("FDLMS", "FDNLMS"):
                    x_block = np.stack(x_buffer, axis=1)
                    e_block = np.array(error_buffer)
                    self.W += self.update_step(e_n=e_block, x_n=x_block)

        # taking clock-time before running metrics
        elapsed_time = time.perf_counter() - start_time

        # to avoid memory issues, need to ensure signals are same shape
        d_flat, y_flat, clean_flat, error_flat = evaluation_runner.signal_reshaper(
            d, noise_estimate, clean_signal, error
        )

        # What algo minimized
        adaption_mse_result = evaluation_runner.MSE(d_flat, y_flat)
        # How close e[n] is to s[n]
        speech_mse_result = evaluation_runner.MSE(clean_flat, error_flat)
        # full SNR
        snr_result = evaluation_runner.SNR(clean_flat, error_flat)
        # # getting the Delta SNR
        snr_in = evaluation_runner.SNR(clean_flat, d_flat)  # SNR without filtering
        delta_snr = snr_result - snr_in

        # NOTE: need to return MSE history...
        return (
            error,
            noise_estimate,
            adaption_mse_result,
            speech_mse_result,
            snr_result,
            delta_snr,
            elapsed_time,
        )


# for frequency domain block based processing
class FrequencyDomainAF(BlockFilterModel):
    def __init__(
        self,
        mu: float,
        filter_order: int,
        block_size: int,
    ) -> None:
        self.mu = mu
        self.N = filter_order
        self.block_size = block_size
        self.hop_size = block_size // 2
        # Init block base
        super().__init__(mu=mu, filter_order=filter_order, block_size=block_size)

        # Overriding with FD specs
        self.half_bins = self.block_size // 2 + 1
        self.eps = 1e-8
        self.algorithm = "FDAF"
        self.stft = STFT(window_length=block_size)
        self.W = np.zeros(self.half_bins, dtype=np.complex128)

    # Setting the update step to include conj
    def update_step(self, e_f: NDArray[Any], x_f: NDArray[Any]) -> NDArray[Any]:
        """Update for FDAF.

        Args:
            e_f (NDArray[np.complex128]): Block error in the frequency domain.
            x_f (NDArray[np.complex128]): Block noise estimate in the frequency domain.

        Returns:
            NDArray[np.float64]: The weight update vector for FDAF.
        """
        return self.mu * np.multiply(np.conj(x_f), e_f)

    # FD specs
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

        Raises:
            ValueError: If Signal dims are not compatible (1D)
        """
        # turning D and X into np arrays, if not already
        d = np.asarray(d).ravel()
        if d.ndim != 1:
            raise ValueError(f"Expected desired signal to be 1D, got shape: {d.shape}")

        x = np.asarray(x).ravel()
        if x.ndim != 1:
            raise ValueError(f"Expected input signal to be 1D, got shape: {x.shape}")

        clean_signal = np.asarray(clean_signal).ravel()
        if clean_signal.ndim != 1:
            raise ValueError(
                f"Expected clean signal to be 1D, got shape: {clean_signal.shape}"
            )

        if d.shape[0] < x.shape[0]:
            x = x[: d.shape[0]]
            # assert x.shape == d.shape  # Double check
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

        # creating an array to track MSE history for convergence metrics
        mse_history = np.zeros(num_samples)

        # creating a ciruclar buffer for the filter taps
        circ_buffer = np.zeros(self.N, dtype=float)
        desired_circ_buffer = np.zeros(self.N, dtype=float)
        # for buffering blocks
        x_buffer = deque(maxlen=self.block_size)
        error_buffer = deque(maxlen=self.block_size)

        # clock-time for how long filtering this signal takes
        start_time = time.perf_counter()
        for sample in range(num_samples):
            # using a circular buffer style window technique:
            circ_buffer = np.roll(circ_buffer, 1)
            desired_circ_buffer = np.roll(desired_circ_buffer, 1)
            # writer-pointer to add the most recent sample into the N buffer window
            circ_buffer[0] = x[sample]
            desired_circ_buffer[0] = d[sample]
            # Buffering blocks
            x_buffer.append(x[sample])
            # error_buffer.append(error[sample])

            # FD update,  hop = %50 block_size
            if len(x_buffer) == (self.block_size) and (sample % self.hop_size) == 0:
                # STFT
                x_f = self.stft.stft(np.array(x_buffer))
                d_f = self.stft.stft(
                    np.array(d[sample - self.block_size + 1 : sample + 1])
                )
                # now in F domain, noise estimate is just mult.
                y_f = self.W * x_f[0]
                # error easily is DF - YF
                e_f = d_f[0] - y_f
                # now updating with FD in mind
                self.W += self.update_step(e_f=e_f, x_f=x_f[0])
                # appending time domain of yf and ef
                start = sample - self.block_size + 1
                end = sample + 1
                noise_estimate[start:end] = self.stft.istft(y_f[np.newaxis, :])
                error[start:end] = self.stft.istft(e_f[np.newaxis, :])

        # taking clock-time before running metrics
        elapsed_time = time.perf_counter() - start_time

        # to avoid memory issues, need to ensure signals are same shape
        d_flat, y_flat, clean_flat, error_flat = evaluation_runner.signal_reshaper(
            d, noise_estimate, clean_signal, error
        )

        # What algo minimized
        adaption_mse_result = evaluation_runner.MSE(d_flat, y_flat)
        # How close e[n] is to s[n]
        speech_mse_result = evaluation_runner.MSE(clean_flat, error_flat)
        # full SNR
        snr_result = evaluation_runner.SNR(clean_flat, error_flat)
        # # getting the Delta SNR
        snr_in = evaluation_runner.SNR(clean_flat, d_flat)  # SNR without filtering
        delta_snr = snr_result - snr_in

        # NOTE: need to return MSE history...
        return (
            error,
            noise_estimate,
            adaption_mse_result,
            speech_mse_result,
            snr_result,
            delta_snr,
            elapsed_time,
        )
