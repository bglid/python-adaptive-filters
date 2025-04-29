# ADAPTIVE LINE ENHANCER UTIL
import numpy as np
from numpy.typing import NDArray


def adaptive_line_enhancer(
    d: NDArray[np.float64], ale_delay: int
) -> NDArray[np.float64]:
    """...

    Args:
        d (NDArray[np.float64]):
            "Desired Signal", which in the ANC use-case is the noisy input signal.
        ale_delay (int): Amount of delay for Adaptive Line Enhancer in samples.

    Returns:
        x (NDArray[np.float64]):
            Input reference matrix X, created from delaying the noisy signal.

    """
    return np.concatenate((np.zeros(ale_delay), d[:-ale_delay]))
