#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

"""
PLOTTING UTILS FOR PRESENTING DIFFERNT ALGORITHMS
"""


# plot to measure time domain of signal
def signal_plot(signal, fs):
    """
    Summary: Plot to trace a given signal in the time domain

    """

    # indexing x to get points over time
    # x_time = np.arange(start=0, stop=signal.shape[0], step=fs)
    # plt.plot(x_time, signal[x_time])
    plt.plot(signal, color="darkviolet")
    plt.grid(True)
    plt.show()


# plot to measure error changes over time


# plot to capture the weight changes over time

fs = 44100
# wave_buffer = np.zeros(shape=(100))
time = np.arange(0, 1.0, step=(1 / fs))
sine = 2 * np.sin(2 * np.pi * 4 * time)

signal_plot(sine, fs)
