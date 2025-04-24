from typing import Any

import csv
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
        # NOTE: for checking the class
        print(filter.__class__)

    return filter


# for demonstrating the results of a given algorithm
def noise_evaluation(
    filter_order: int,
    mu: float,
    algorithm: str,
    noise: str,
    snr_levels: int = 5,
    save_result: bool = False,
    eval_at_sample: int = 1000,
) -> dict[str, float]:
    """Iterates Adaptive filter algorithm over full dataset of noise, returning Average result. NOTE: Files must be presorted

    Args:
        filter_order (int): The n-value of the filter window size.
        mu (float): Learning rate parameter (Mu).
        algorithm (str): Adaptive Filter Algorithm to be used.
        noise (str): Which noise type with be evaluated. Points to directories containing all related data.
        snr_levels (int): How many differing SNR levels are tested. Default is 5.
        save_result (bool): Whether or not individual .wav files and plots should be written or saved. Default is False.
        eval_at_sample (int): At n samples an evaluation should be taken. Smaller values mean more evaluations.

    Returns:
        dict: Dictionary of mean result for each metric for provided noise set.
    """
    mean_results: dict[str, float] = {}

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

    # taking the mean of the metrics for this noise
    mean_results[f"{algorithm} Mean MSE: {noise} noise"] = np.mean(all_mse)
    mean_results[f"{algorithm} Mean SNR: {noise} noise"] = np.mean(all_snr)
    print("Checking mean results")
    print(mean_results)

    return mean_results


def full_evaluation(
    filter_order: int,
    mu: float,
    algorithm: str,
    noise: str,
    snr_levels: int = 5,
    save_result: bool = False,
    eval_at_sample: int = 1000,
) -> dict[str, dict[str, float]]:
    """Runs the evaluation aggregated evalutaion metrics for each noise type provided. Writes final results to a .csv.

    Args:
        filter_order (int): The n-value of the filter window size.
        mu (float): Learning rate parameter (Mu).
        algorithm (str): Adaptive Filter Algorithm to be used.
        noise (str): String of noise type to run evaluation metrics on. If all, runs every metric
        snr_levels (int): How many differing SNR levels are tested. Default is 5.
        save_result (bool): Whether or not individual .wav files and plots should be written or saved. Default is False.
        eval_at_sample (int): At n samples an evaluation should be taken. Smaller values mean more evaluations.

    Returns:
        dict: Dictionary of dictionaries, each containing the metrics for a given noise type.
    """
    # list of valid noises that can work here
    valid_noise = [
        "air_conditioner",
        "babble",
        "cafe",
        "munching",
        "typing",
        "washer_dryer",
    ]
    # creating a dictionary to hold the results for each noise type
    final_results: dict[str, dict[str, float]] = {}
    # check if 'all' is passed as a noise type
    if noise == "all":
        # Run the evaluation for each noise type
        for i in range(len(valid_noise)):
            # getting results from noise eval function
            print(
                f"-----Starting testing on {valid_noise[i]} with the {algorithm} algorithm-----"
            )
            # passing valid noise, which is the noise at this iter
            result = noise_evaluation(
                filter_order,
                mu,
                algorithm,
                valid_noise[i],
                snr_levels,
                save_result,
                eval_at_sample,
            )

            final_results[f"{algorithm} All {valid_noise[i]} results "] = result
            print(f"Run resuls: \t{result}")

            # writing the results to a csv to the data directory
            fields = [
                f"{algorithm} Mean MSE: {valid_noise[i]} noise",
                f"{algorithm} Mean SNR: {valid_noise[i]} noise",
            ]
            print(fields[0])
            print(fields[1])
            # writing each
            with open(
                f"./data/tabular_results/{algorithm}/{valid_noise[i]}_results.csv",
                "w",
                newline="",
            ) as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        fields[0]: result[fields[0]],
                        fields[1]: result[fields[1]],
                    }
                )
            # logging feedback that noise type was written
            print(f"{valid_noise[i]} type written!")
            print(f"Results: \n{result[fields[0]]} \n{result[fields[1]]}")
            print(f"----------------------------------------")

        print(f" FINAL RESULTS!!! \n{final_results}")
        return final_results

    # else, we need to give feedback that input is incorrect
    else:
        print("For running this script, it's better to run on all inputs")
        print("To do so, pass 'all' as noise...")


if __name__ == "__main__":

    full_evaluation(32, 0.01, "LMS", "all")
    # noise_evaluation(32, 0.01, "LMS", "air_conditioner")
