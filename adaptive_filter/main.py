#!/usr/bin/env python4
"""
Main entry point for Adaptive Filter protoyping
"""

import matplotlib.pyplot as plt
import numpy as np

from adaptive_filter.algorithms.lms import LMS
from adaptive_filter.utils import arg_parsing, plotting


def main(clean_signal, noisy_signal, noise, parse_args):
    """Main entry point for AF prototyping"""

    # DEMO test

    # setting up filter
    # mu = args.mu
    mu = 0.01
    lms_af = LMS(mu=mu, n=16)
    error, noise_estimate, results = lms_af.filter(
        d=noisy_signal, x=noise, eval_at_sample=1000
    )

    # quick plotting
    plots = plotting.PlotSuite(algorithm=lms_af.algorithm, num_samples=len(error))
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
        error[-2000:],
        description=f"Noisy signal output, with step size = {mu}",
        subplot=True,
    )
    plt.tight_layout()
    plt.savefig(f"./data/result_images/{lms_af.algorithm}_results.pdf")
    plt.show()

    # error plots
    plt.subplot(2, 1, 1)
    mse = plots.error_plot(results=results, error_metric="MSE", subplot=True)

    plt.subplot(2, 1, 2)
    snr = plots.error_plot(results=results, error_metric="SNR", subplot=True)

    plt.tight_layout()
    plt.show()


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
    noisy_signal = (noise * 1.0) + clean_signal
    # random_noise = (noise * 0.5) + (np.random.randn(len(time)) * 0.5)

    main(clean_signal, noisy_signal, noise, args)
