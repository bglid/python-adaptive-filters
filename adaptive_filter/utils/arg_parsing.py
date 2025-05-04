"""
Utils for parsing args that allow easy experimentation
and testing of different algorithms and params from CLI
"""

import argparse


def parse_args():

    parser = argparse.ArgumentParser(
        description="For demo and experimenting with different DSP AFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--eval",
        type=bool,
        choices=[True, False],
        help="Choice to run project as evaulation over MS-SNSD data.",
    )
    parser.add_argument(
        "--filter_order",
        type=int,
        default=16,
        help="Filter order for Adaptive DSP filter",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.01,
        help="Step size for Adaptive Filter learning rate",
    )

    # choosing which algorithm to test
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["LMS", "NLMS", "RLS", "APA", "FDLMS", "FDNLMS"],
        help="Choice of Adaptive Filter Algorithm to test",
    )

    parser.add_argument(
        "--noise",
        type=str,
        choices=[
            "all",
            "air_conditioner",
            "babble",
            "cafe",
            "munching",
            "typing",
            "washer_dryer",
        ],
        help="Choice of which noise type to test filter on. If 'all' is selected, runs on every type",
    )
    parser.add_argument(
        "--delay_amount",
        type=float,
        default=0.0,
        help="Amount of extra delay to add to noise reference.",
    )
    parser.add_argument(
        "--rand_noise",
        type=int,
        default=40,
        help="Amount of extra white noise to add to noise reference.",
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=16000,
        help="Sampling Rate - defaults to 16 kHz.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=0,
        help="Block size of filter, used in filters that inherit from the Block-based class such as APA and FD.",
    )
    parser.add_argument(
        "--snr_levels",
        type=int,
        default=0,
        help="Amount of variation in snr levels.",
    )
    parser.add_argument(
        "--ale",
        type=bool,
        default=False,
        choices=[True, False],
        help="Whether to run evaluation with noise reference (False), or using Adaptive Line Enhancer (True)",
    )
    parser.add_argument(
        "--ale_delay",
        type=int,
        default=0,
        help="Amount in samples that ALE should delay to create a noise reference.",
    )

    parser.add_argument(
        "--save_result",
        type=bool,
        default=True,
        choices=[True, False],
        help="Choice of whether to save resulting sound and image file",
    )

    return parser.parse_args()
