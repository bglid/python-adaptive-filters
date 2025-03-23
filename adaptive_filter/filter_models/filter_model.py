# Class that contains filter model used by most adaptive filters
import numpy as np
from numpy.typing import NDArray


class FilterModel:
    def __init__(self, mu: float, n: int) -> None:
        # consider adding p: order
        self.mu = mu  # step_rate
        self.N = n  # filter window size

    def predict_y(self, x_n: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predicts the output y[n], given vector X[n]. Uses formula W^T[n]X[n]

        Args:
            x_n (np.ndarray): vector[n] of array X

        Returns:
            np.float64: Predicted output at n
        """
        return np.dot(self.W, x_n)

    def error(self, d_n: float, y_n: float) -> float:
        """Calculates the error, e[n] = d[n] - y[n], y[n] is output of W^T[n]X[n]

        Args:
            d_n (float): Desired sample at point n of array D
            y_n (float): Prediction of y[n]

        Returns:
            float: error of desired input[n] - predicted input (y[n])
        """
        return d_n - y_n

    def update_step(self, e_n: float, x_n: NDArray[np.float64]) -> NDArray[np.float64]:
        """Updates weights of W[n + 1], given the learning algorithm chosen

        Args:
            e_n (float): Error sample at point n
            x_n (np.ndarray): Input vector n

        Returns:
            np.ndarray: Update step to self.W
        """
        return np.zeros(len(x_n))

    def filter(self, d, x):
        """Iterates Adaptive filter alorithm and updates for length of input signal X

        Args:
            d (np.ndarray): Desired Vector array D
            x (np.ndarray): Input matrix X

        Returns:
            tuple: A tuple containing:
                - np.ndarray: Predicted output signal.
                - np.ndarray: The error signal of d -y.
        """

        # initializing our weights given X
        self.W = np.random.normal(0.0, 0.5, self.N)
        # getting the number of samples from x len
        num_samples = len(x)

        # turning D and X into np arrays, if not already
        if type(d) is not NDArray:
            d = np.array(d)

        if type(x) is not NDArray:
            x = np.array(x)

        print(x.shape)

        # initializing the arrays to hold error and predictions
        y = np.zeros(num_samples)
        error = np.zeros(num_samples)
        # creating an array to track the weight changes over time N
        # self.weight_t = np.zeros(())

        for sample in range(num_samples):
            # getting the prediction y
            y[sample] = self.predict_y(x[sample])
            # getting the error e[sample] = d[sample] - y[sample]
            error[sample] = self.error(d_n=d[sample], y_n=y[sample])
            # updating the weights
            self.W += self.update_step(e_n=error[sample], x_n=x[sample])

        return y, error
