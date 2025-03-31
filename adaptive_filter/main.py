#!/usr/bin/env python3
"""
Main entry point for Adaptive Filter protoyping
"""

import numpy as np

from adaptive_filter.algorithms.lms import LMS
from adaptive_filter.utils import arg_parsing, plotting


def main(clean_signal, noisy_signal, args):
    """Main entry point for AF prototyping"""

    # DEMO test

    # setting up filter
    # mu = args.mu
    mu = 0.6
    lms_af = LMS(mu=mu, n=noisy_signal[0].shape)
    y, error = lms_af.filter(d=clean_signal, x=noisy_signal)
    # print(f"y: {y}")
    # print(f"\n Error:{error}")
    #
    # print(f"Size: {y.shape}")

    # plotting the normal signal
    plotting.signal_plot(clean_signal[998000:])

    # plotting the noisy signal
    plotting.signal_plot(noisy_signal[998000:])

    # plotting the new signal!
    plotting.signal_plot(y[998000:])
    # plotting.signal_plot(error)


if __name__ == "__main__":

    args = arg_parsing.parse_args()

    # clean signal
    fs = 1000
    frequency = 5
    time = np.arange(0.0, 1000.0, step=(1 / fs))
    clean_signal = np.sin(2 * np.pi * frequency * time)

    # creating noise to create the noisy signal
    noise = np.random.randn(len(time))
    # creating the noisy signal
    noisy_signal = (noise * 0.4) + clean_signal

    main(clean_signal, noisy_signal, args)
