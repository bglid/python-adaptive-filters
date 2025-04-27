import numpy as np
from numpy.typing import NDArray

from adaptive_filter.filter_models.block_filter_model import BlockFilterModel


class FD_LMS(BlockFilterModel):
    def __init__(self, mu: float, n: int, block_size: int) -> None:
        # initializing BlockFilterModel
        super().__init__(mu=mu, filter_order=n, block_size=block_size)
        self.algorithm = "FD_LMS"
        self.eps = 1e-6

    # updating the update step for LMS algorithm
    def update_step(self, e_n: float, x_n: NDArray[np.float64]) -> NDArray[np.float64]:
        """Update for FDAF:

        Args:
            e_n (float): Error sample (n)
            x_n (NDArray[np.float64]): vector[n] of array X, the noise estimate

        Returns:
            NDArray[np.float64]: The weight update vector for FDAF.
        """
        return np.array([0.0])
