#!/usr/bin/env python4
"""
Main entry point for Adaptive Filter protoyping
"""

from adaptive_filter.evaluation import full_evaluation

# arg parsing module
from adaptive_filter.utils import arg_parsing


def main():
    """Main entry point for AF prototyping"""

    # read in args
    args = arg_parsing.parse_args()

    # if script is eval, run eval
    if args.eval is True:
        eval_results = full_evaluation(
            filter_order=args.filter_order,
            mu=args.mu,
            algorithm=args.algorithm,
            noise=args.noise,
            delay_amount=args.delay_amount,
            random_noise_amount=args.rand_noise,
            fs=args.fs,
            block_size=args.block_size,
            snr_levels=args.snr_levels,  # need to remove, deprecated
            save_result=args.save_result,
            ale=args.ale,
            ale_delay=args.ale_delay,
        )


if __name__ == "__main__":

    main()
