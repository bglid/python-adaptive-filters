#!/usr/bin/env python4
"""
Main entry point for Adaptive Filter protoyping
"""

import matplotlib.pyplot as plt
import numpy as np

from adaptive_filter.algorithms.lms import LMS
from adaptive_filter.utils import arg_parsing, plotting


def main(clean_signal, noisy_signal, args):
    """Main entry point for AF prototyping"""

    # DEMO test

    # setting up filter
    # mu = args.mu
    mu = 0.44
    lms_af = LMS(mu=mu, n=16)
    y, error, results = lms_af.filter(
        d=clean_signal, x=noisy_signal, eval_at_sample=5000
    )

    # quick plotting
    plots = plotting.PlotSuite(algorithm=lms_af.algorithm, num_samples=len(y))
    plt.subplot(3, 1, 1)
    clean_plot = plots.signal_plot(
        clean_signal[-2000:], description="Clean signal input", subplot=True
    )

    plt.subplot(3, 1, 2)
    noisy_plot = plots.signal_plot(
        noisy_signal[-2000:], description="Noisy signal input", subplot=True
    )

    plt.subplot(3, 1, 3)
    output_plot = plots.signal_plot(
        y[-2000:],
        description=f"Noisy signal output, with step size = {mu}",
        subplot=True,
    )
    plt.tight_layout()
    plt.savefig(f"./data/result_images/{lms_af.algorithm}_results.pdf")
    plt.show()

    # # print(len(results["MSE"]))
    # plt.figure(figsize=(10, 6))
    # x_axis = np.arange(0, len(results["MSE"]))
    # plt.plot(x_axis, results["MSE"], c="darkviolet")
    # plt.title("MSE")
    # plt.grid()
    # plt.show()
    #
    # plt.figure(figsize=(10, 6))
    # x_axis = np.arange(0, len(results["SNR"]))
    # plt.plot(x_axis, results["SNR"], c="crimson")
    # plt.title("SNR")
    # plt.grid()
    # plt.show()


if __name__ == "__main__":

    args = arg_parsing.parse_args()

    # clean signal
    fs = 1000
    frequency = 5
    time = np.arange(0.0, 1000.0, step=(1 / fs))
    clean_signal = 2 * np.sin(2 * np.pi * frequency * time)

    # creating noise to create the noisy signal
    noise = np.random.randn(len(time))
    # creating the noisy signal
    noisy_signal = (noise * 0.3) + clean_signal

    main(clean_signal, noisy_signal, args)
