from typing import Any

import glob

import librosa
import numpy as np

from adaptive_filter.algorithms import apa, frequency_domain, fx_lms, lms, nlms, rls
from adaptive_filter.filter_models.filter_model import FilterModel


# function for loading the data for evaluation
def load_data(noise: str, snr_levels: int = 5) -> tuple:
    """Reads in data into three different lists for running evaluation.

    Args:
        noise (str): Which noise type with be evaluated.
        snr_levels (int): How many differing SNR levels are tested. Default is 5.

    Returns:
        tuple: Lists of each loaded dataset:
            - List[Any]: List of the noise .wavs
            - List[Any]: List of the noisy speech .wavs
            - List[Any]: List of the clean speech .wavs
    """
    # creating variables to hold the read paths
    read_paths: dict[str, Any] = {
        "Noise": f"./data/evaluation_data/{noise}/Noise_training/",
        "Noisy_Speech": f"./data/evaluation_data/{noise}/NoisySpeech_training/",
        "Clean_Speech": f"./data/evaluation_data/{noise}/CleanSpeech_training/",
    }
    # Loading all the noise files
    noise_wavs = []
    for file in glob.iglob(f"{read_paths['Noise']}/*"):
        noise_file, noise_sr = librosa.load(file, sr=None)
        noise_wavs.append(noise_file)
        # print(noise_file.shape)
    # Turning into np array
    noise_array = np.array(noise_wavs, dtype=object)

    # Loading all the noisy_speech files
    noisy_speech_wavs = []
    for file in glob.iglob(f"{read_paths['Noisy_Speech']}/*"):
        noisy_speech_file, noisy_speech_sr = librosa.load(file, sr=None)
        noisy_speech_wavs.append(noisy_speech_file)
    # Turning into np array
    noisy_speech_array = np.array(noisy_speech_wavs, dtype=object)

    # Loading all the clean_speech files
    clean_speech_wavs = []
    for file in glob.iglob(f"{read_paths['Clean_Speech']}/*"):
        # appending extra copies of the speech data depending on SNR level
        clean_speech_file, clean_speech_sr = librosa.load(file, sr=None)
        for j in range(snr_levels):
            clean_speech_wavs.append(clean_speech_file)
    # Turning into np array
    clean_speech_array = np.array(clean_speech_wavs, dtype=object)

    # return noise_wavs, noisy_speech_wavs, clean_speech_wavs
    return noise_array, noisy_speech_array, clean_speech_array


# function for selecting the filter
def select_algorithm(
    filter_order: int,
    mu: float,
    algorithm: str,
) -> FilterModel:
    """Selects and returns instance of FilterModel based on algorithm passed

    Args:
        filter_order (int): The n-value of the filter window size.
        mu (float): Learning rate parameter (Mu).
        algorithm (str): Adaptive Filter Algorithm to be used.

    Returns:
        FilterModel: Instance of filter based on Algorithm

    Raises:
        ValueError: If 'algorithm' is not one of the supported algorithms.
    """
    # setting filter algorithm via dict
    algos: dict[str, FilterModel] = {
        "LMS": lms.LMS(mu=mu, n=filter_order),
    }
    # filter algorithm is defined by input
    # checking first that input isn't faulty
    if algorithm not in algos:
        raise ValueError(f"Uknown algorithm: '{algorithm}'... ")

    if algorithm is not None:
        # setting filter with given inputs
        filter = algos[algorithm]
        print(filter.__class__)

    return filter


# for demonstrating the results of a given algorithm
def run_evaluation(
    filter_order: int,
    mu: float,
    algorithm: str,
    noise: str,
    snr_levels: int = 5,
    save_result: bool = False,
    eval_at_sample: int = 1000,
) -> None:
    """Iterates Adaptive filter algorithm over full dataset, returning Average result. NOTE: Files must be presorted

    Args:
        filter_order (int): The n-value of the filter window size.
        mu (float): Learning rate parameter (Mu).
        algorithm (str): Adaptive Filter Algorithm to be used.
        noise (str): Which noise type with be evaluated. Points to directories containing all related data.
        snr_levels (int): How many differing SNR levels are tested. Default is 5.
        save_result (bool): Whether or not individual .wav files and plots should be written or saved. Default is False.
        eval_at_sample (int): At n samples an evaluation should be taken. Smaller values mean more evaluations.
    """

    # Loading data from respective data paths
    noise_list, noisy_speech_list, clean_speech_list = load_data(noise)

    # getting filter algorithm
    af_filter = select_algorithm(filter_order, mu, algorithm)

    # Allocating arrays for mse and snr results to average
    all_mse = np.zeros(shape=noise_list.shape[0])
    all_snr = np.zeros(shape=noise_list.shape[0])

    # Run the filtering algorithm per instance of noise
    for i in range(noise_list.shape[0]):
        error, noise_estimate, results, mse_i, snr_i = af_filter.filter(
            d=noisy_speech_list[i],
            x=noise_list[i],
            clean_signal=clean_speech_list[i],
            eval_at_sample=1000,
            weighted_evaluation=True,
        )
        # appending mean mse and snr to later avg.
        all_mse[i] = mse_i
        all_snr[i] = snr_i

    print(all_mse.shape)
    print(all_snr.shape)


if __name__ == "__main__":

    run_evaluation(16, 0.01, "LMS", "air_conditioner")
