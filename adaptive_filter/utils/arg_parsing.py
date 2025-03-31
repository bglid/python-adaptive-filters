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
    # setting up subparsers for the different filter algorithms
    # subparsers = parser.add_subparsers(help="Types of filter algorithms")

    parser.add_argument(
        "--filter_order",
        type=int,
        default=4,
        help="Filter order for Adaptive DSP filter",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.5,
        help="Step size for Adaptive Filter learning rate",
    )

    # choosing which algorithm to test
    # NOTE: May want to adjust depending on future algorithms specifications
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["LMS", "RLS"],
        help="Choice of Adaptive Filter Algorithm to test",
    )

    return parser.parse_args()
