import numpy as np
from numpy.typing import NDArray

from adaptive_filter.filter_models.block_filter_model import BlockFilterModel


class FD_NLMS(BlockFilterModel):
    def __init__(self, mu: float, n: int) -> None:
        self.mu = mu
        self.N = n
        self.algorithm = "FD_NLMS"
        self.eps = 1e-6

    # updating the update step for FD_NLMS algorithm
    def update_step(self, e_n: float, x_n: NDArray[np.float64]) -> NDArray[np.float64]:
        """Update for FDAF:

        Args:
            e_n (float): Error sample (n)
            x_n (NDArray[np.float64]): vector[n] of array X, the noise estimate

        Returns:
            NDArray[np.float64]: The weight update vector for FDAF.
        """
        return np.array([0.0])
