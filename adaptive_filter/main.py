#!/usr/bin/env python4
"""
Main entry point for Adaptive Filter protoyping
"""

# import librosa
# import matplotlib.pyplot as plt
# import numpy as np
# import soundfile as sf

# from adaptive_filter.algorithms.lms import LMS

from adaptive_filter.evaluation import run_evaluation

# arg parsing module
from adaptive_filter.utils import arg_parsing


def main():
    """Main entry point for AF prototyping"""

    # read in args
    args = arg_parsing.parse_args()

    # if script is eval, run eval

    # NOTE: Below three are for individual results
    # quick plotting
    # plots = plotting.PlotSuite(algorithm=lms_af.algorithm, num_samples=len(error))
    # plt.subplot(3, 1, 1)
    # clean_plot = plots.signal_plot(
    #     clean_signal, description="Clean signal input", subplot=True
    # )
    # plt.subplot(3, 1, 2)
    # noisy_plot = plots.signal_plot(
    #     noisy_signal, description="Noisy signal input", subplot=True
    # )
    # plt.subplot(3, 1, 3)
    # output_plot = plots.signal_plot(
    #     error,
    #     description=f"Noisy signal output, with step size = {mu}",
    #     subplot=True,
    # )
    # plt.tight_layout()
    # plt.savefig(f"./data/result_images/{lms_af.algorithm}_results.pdf")
    # plt.show()

    # NOTE: Below two are for individual results
    # # error plots
    # plt.subplot(2, 1, 1)
    # mse = plots.error_plot(results=results, error_metric="MSE", subplot=True)
    # plt.subplot(2, 1, 2)
    # snr = plots.error_plot(results=results, error_metric="SNR", subplot=True)
    # plt.tight_layout()
    # plt.show()

    # NOTE: Below is for saving individual results
    # # testing the result
    # sf.write("./data/processed_data/test_1_clean.wav", clean_signal, samplerate=16000)
    # sf.write("./data/processed_data/test_1_noisy.wav", noisy_signal, samplerate=16000)
    # sf.write("./data/processed_data/test_1.wav", error, samplerate=16000)


if __name__ == "__main__":

    # RUNNING SOME ACTUAL TESTS!!!
    # clean_speech_load = librosa.load(
    #     "./data/evaluation_data/air_conditioner/CleanSpeech_training/clnsp1.wav",
    #     sr=None,
    # )
    # noisy_speech_load = librosa.load(
    #     "./data/evaluation_data/air_conditioner/NoisySpeech_training/noisy1_SNRdb_0.0_clnsp1.wav",
    #     sr=None,
    # )
    # noise_load = librosa.load(
    #     "./data/evaluation_data/air_conditioner/Noise_training/noisy1_SNRdb_0.0.wav",
    #     sr=None,
    # )
    # clean_speech = clean_speech_load[0]
    # noisy_speech = noisy_speech_load[0]
    # noise = noise_load[0]

    main()
