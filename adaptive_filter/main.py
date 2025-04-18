#!/usr/bin/env python4
"""
Main entry point for Adaptive Filter protoyping
"""

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# quick ICA test
from sklearn.decomposition import NMF, FastICA

from adaptive_filter.algorithms.lms import LMS
from adaptive_filter.utils import arg_parsing, plotting


def main(clean_signal, noisy_signal, noise, parse_args):
    """Main entry point for AF prototyping"""

    # DEMO test

    # setting up filter
    # mu = args.mu
    mu = 0.005
    lms_af = LMS(mu=mu, n=64)
    error, noise_estimate, results = lms_af.filter(
        d=noisy_signal, x=noise, eval_at_sample=1000, clean_signal=clean_signal
    )

    # quick plotting
    plots = plotting.PlotSuite(algorithm=lms_af.algorithm, num_samples=len(error))
    plt.subplot(3, 1, 1)
    clean_plot = plots.signal_plot(
        clean_signal, description="Clean signal input", subplot=True
    )

    plt.subplot(3, 1, 2)
    noisy_plot = plots.signal_plot(
        noisy_signal, description="Noisy signal input", subplot=True
    )

    plt.subplot(3, 1, 3)
    output_plot = plots.signal_plot(
        error,
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

    # testing the result
    sf.write("./data/processed_data/test_1_clean.wav", clean_signal, samplerate=16000)
    sf.write("./data/processed_data/test_1_noisy.wav", noisy_signal, samplerate=16000)
    sf.write("./data/processed_data/test_1.wav", error, samplerate=16000)


if __name__ == "__main__":

    args = arg_parsing.parse_args()

    # # clean signal
    # fs = 1000
    # frequency = 5
    # time = np.arange(0.0, 1000.0, step=(1 / fs))
    # clean_signal = 2 * np.sin(2 * np.pi * frequency * time)
    #
    # # creating noise to create the noisy signal
    # noise = np.random.randn(len(time))
    # # creating the noisy signal
    # noisy_signal = (noise * 1.5) + clean_signal
    # # random_noise = (noise * 0.5) + (np.random.randn(len(time)) * 0.5)

    # RUNNING SOME ACTUAL TESTS!!!
    clean_speech = librosa.load(
        "./data/evaluation_data/air_conditioner/CleanSpeech_training/clnsp1.wav",
        sr=None,
    )
    noisy_speech = librosa.load(
        "./data/evaluation_data/air_conditioner/NoisySpeech_training/noisy1_SNRdb_0.0_clnsp1.wav",
        sr=None,
    )
    noise = librosa.load(
        "./data/evaluation_data/air_conditioner/Noise_training/noisy1_SNRdb_0.0.wav",
        sr=None,
    )
    clean_speech = clean_speech[0]
    noisy_speech = noisy_speech[0]
    noise = noise[0]

    main(clean_speech, noisy_speech, noise, args)
