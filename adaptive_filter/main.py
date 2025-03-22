#!/usr/bin/env python3
"""
Main entry point for Adaptive Filter protoyping
"""

import argparse
import sys

import numpy as np

from adaptive_filter.algorithms.lms import LMS


def main():
    """Main entry point for AF prototyping"""
    # random demo data
    N = 500
    x = np.random.normal(0, 1, (N, 4))
    v = np.random.normal(0, 0.1, N)
    d = 2 * x[:, 0] + 0.1 * x[:, 1] - 4 * x[:, 2] + 0.5 * x[:, 3] + v

    # setting up filter
    mu = 0.5
    lms_af = LMS(mu=mu, n=4)
    y, error = lms_af.filter(d=d, x=x)
    print(f"y: {y}")
    print(f"\n Error:{error}")

    print(f"Size: {y.shape}")


if __name__ == "__main__":

    main()
