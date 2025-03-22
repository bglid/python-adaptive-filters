# Class that contains filter model used by most adaptive filters
import numpy as np


class FilterModel:
    def __init__(self, mu: float) -> None:
        # consider adding p: order
        self.mu = mu  # step_rate

    def predict_y(self, x_n: np.ndarray) -> float:
        """Predicts the output y[n], given vector X[n]. Uses formula W^T[n]X[n]

        Args:
            x_n ndarray: vector[n] of array X

        Returns:
            y_n float: Predicted output at n
        """
        return self.W.T @ x_n

    def error(self, d_n: float, y_n: float) -> float:
        """Calculates the error, e[n] = d[n] - y[n], y[n] is output of W^T[n]X[n]

        Args:
            d_n float: Desired sample at point n of array D
            y_n float: Prediction of y[n]

        Returns:
            error_n float: error of desired input[n] - predicted input (y[n])
        """
        return d_n - y_n

    def update_step(self, e_n: float, x_n: np.ndarray):
        """Updates weights of W[n + 1], given the learning algorithm chosen

        Args:
            e_n float: Error sample at point n
            x_n ndarray: Input vector n

        Returns:
            Update step to self.W
        """
        return np.zeros(len(x_n))

    def filter(self, d, x):
        """Iterates Adaptive filter alorithm and updates for length of input signal X

        Args:
            d ndarray: Desired Vector array D
            x ndarray: Input matrix X

        Returns:
        """

        # getting the filter window, N, from x len
        self.N = len(x)

        # turning D and X into np arrays, if not already
        if type(d) is not np.ndarray:
            d = np.array(d)

        if type(x) is not np.ndarray:
            x = np.array(x)

        # initializing our weights given X
        self.W = np.random.normal(0.0, 0.5, self.N)

        # initializing the arrays to hold error and predictions
        y = np.zeros(self.N)
        error = np.zeros(self.N)
        # creating an array to track the weight changes over time N
        # self.weight_t = np.zeros(())

        for sample in range(self.N):
            # getting the prediction y
            y[sample] = self.predict_y(x[sample])
            # getting the error e[sample] = d[sample] - y[sample]
            error[sample] = self.error(d_n=d[sample], y_n=y[sample])
            # updating the weights
            self.W += self.update_step(e_n=error[sample], x_n=x[sample])

        return y, error
