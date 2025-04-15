# Class that contains filter model used by most adaptive filters
import numpy as np
from numpy.typing import NDArray

from adaptive_filter.utils.evaluation import EvaluationSuite


class FilterModel:
    def __init__(self, mu: float, n: int) -> None:
        # consider adding p: order
        self.mu = mu  # step_rate
        self.N = n  # filter window size
        # Algorithm type, defined by subclass algorithm
        self.algorithm = ""

    def noise_estimate(self, x_n: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predicts the noise estimate, given vector X[n], noise reference. Uses formula W^T[n]X[n]

        Args:
            x_n (np.ndarray): vector[n] of array X, the noise estimate

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
            x_n (np.ndarray): Input vector n

        Returns:
            np.ndarray: Update step to self.W
        """
        return np.zeros(len(x_n))

    def filter(self, d, x, eval_at_sample=100):
        """Iterates Adaptive filter alorithm and updates for length of input signal X

        Args:
            d (np.ndarray): "Desired Signal", which in the ANC use-case is the noisy input signal.
            x (np.ndarray): Input reference matrix X, which in the ANC case is the noise reference.
            eval_at_sample (int): Number of iterations that must pass in order to log output

        Returns:
            tuple: A tuple containing:
                - np.ndarray: "Clean output" - The error signal of d - y
                - np.ndarray: Predicted noise estimate.
                - np.ndarray: Vector of the results
        """

        # initializing our weights given X
        self.W = np.random.normal(0.0, 0.5, x[0].shape)
        self.W *= 0.001  # setting weights close to zero
        # print(self.W.ndim)
        if self.W.ndim <= 1:
            self.W = self.W.reshape(-1, 1)
            # print(self.W.shape)
        assert self.W.ndim == 2

        # getting the number of samples from x len
        num_samples = len(x)

        # creating evaluation object
        evaluation_runner = EvaluationSuite(algorithm=self.algorithm)
        # results = np.zeros(shape=(num_samples % eval_at_sample))
        results = {"MSE": [], "SNR": []}

        # turning D and X into np arrays, if not already
        if type(d) is not NDArray:
            d = np.array(d)
        # asserting the shape of d
        if d.ndim == 1:
            d = d.reshape(-1, 1)  # making shape (n, 1)
        assert d.ndim == 2

        if type(x) is not NDArray:
            x = np.array(x)
        # checking X shape
        if x.ndim == 1:
            x = x.reshape(-1, 1)  # making shape (n, 1)
        assert x.ndim == 2

        # asserting that x and d have the same shape!!

        # initializing the arrays to hold error and noise estimate
        noise_estimate = np.zeros(num_samples)
        error = np.zeros(num_samples)
        # creating an array to track the weight changes over time N
        # self.weight_t = np.zeros(())

        for sample in range(num_samples):
            # getting the prediction y (noise estimate)
            noise_estimate[sample] = self.noise_estimate(x[sample])
            # getting the error e[sample] = d[sample] - y[sample]
            error[sample] = self.error(
                d_n=d[sample], noise_estimate=noise_estimate[sample]
            )
            # updating the weights
            self.W += self.update_step(e_n=error[sample], x_n=x[sample])

            # running eval suite logging if log criteria met
            assert (
                eval_at_sample >= 0
            ), "Please set eval sample criteria to a number greater than zero if logging is desired, else leave at zero"
            # taking an eval log and appending to results array
            if (sample + eval_at_sample) % eval_at_sample == 0 and sample > 0:
                temp_results = evaluation_runner.evaluation(
                    d[sample], noise_estimate[sample], self.mu, time_k=sample
                )
                results["MSE"].append(temp_results["MSE"])
                results["SNR"].append(temp_results["SNR"])

        return error, noise_estimate, results
