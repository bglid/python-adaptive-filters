#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

"""
PLOTTING UTILS FOR PRESENTING DIFFERNT ALGORITHMS
"""


class PlotSuite:
    def __init__(self, algorithm: str, num_samples: int) -> None:
        """
        Args:
            algorithm (str): String description of the Plot being used
            num_samples (int): Time in terms of number of samples

        """
        self.algorithm = algorithm
        self.N = num_samples

    # plot to measure time domain of signal
    def signal_plot(self, signal: NDArray, description: str, subplot: bool = False):
        """
        Summary: Plot to trace a given signal in the time domain

        Args:
            signal (np.ndarray): Vector of output signal y_hat
            description (str): String description of plot, used in title

        """
        # indexing x to get points over time
        plt.plot(signal, c="darkviolet", label=f"{self.algorithm}")
        plt.title(f"{description} of {self.algorithm} algorithm")
        plt.ylabel("Amplitude")  # rotate this
        plt.xlabel(f"Time in {self.N} samples")
        plt.legend()
        plt.grid()

        # if its not a subplot, show results
        if subplot is False:
            plt.show()
        else:
            return plt

    # plot to measure error changes over time
    def error_plot(self, results: NDArray, error_metric: str, subplot: bool = False):
        """
        Summary: Plot to trace the error over time.

        Args:
            results (np.ndarray): Vector of error results
            error_metric (str): String description of error_metric, used in title
        """
        x_axis = np.arange(0, len(results[error_metric]))
        plt.title(f"{error_metric} of {self.algorithm}")
        plt.plot(x_axis, results[error_metric], c="crimson", label=f"{error_metric}")
        plt.title(error_metric)
        plt.ylabel(error_metric)  # rotate this
        plt.xlabel(f"Time in {self.N} samples")
        plt.legend()
        plt.grid()

        # if its not a subplot, show results
        if subplot is False:
            plt.show()
        else:
            return plt

    # plot to capture the weight changes over time
    def weight_adjustments(self, weights: NDArray, time_k: int, subplot: bool = False):
        """Summary: plots the adjustments to the weights over time k.

        Args:
            weights (np.ndarray): Vector of weights at time_k
            time_k (int): Iteration number k.
        """
        pass

        # # if its not a subplot, show results
        # if subplot is False:
        #     plt.show()
        # else:
        #     return plt

    # plot to run all three and return them as subplots in one, if chosen
    def full_plot_suite(
        self,
        signal: NDArray,
        description: str,
        results: NDArray,
        error_metric: str,
        weights: NDArray,
        time_k: int,
    ):
        """Summary: Plots the entire suite of plots available as subplots.

        Args:
            signal (np.ndarray): Vector of output signal y_hat
            description (str): String description of plot, used in title
            results (np.ndarray): Vector of error results
            error_metric (str): String description of error_metric, used in title
            weights (np.ndarray): Vector of weights at time_k
            time_k (int): Iteration number k.

        Returns:
            matplotlib.pyplot: Full plot containing each available plot as a subplot
        """
        # creating each individual plot as a subplot
        plt.figure(figsize=(14, 8))
        plt.title(f"Results of {self.algorithm} algorithm")
        plt.subplot(3, 1, 1)
        signal_plot = self.signal_plot(signal, description, subplot=True)
        plt.subplot(3, 1, 2)
        error_plot = self.error_plot(results, error_metric, subplot=True)

        # show the full plot
        plt.tight_layout()
        plt.show()
