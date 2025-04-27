import numpy as np
from numpy.typing import NDArray

from adaptive_filter.filter_models.block_filter_model import BlockFilterModel


class APA(BlockFilterModel):
    def __init__(self, mu: float, n: int, block_size: int) -> None:
        # initializing BlockFilterModel
        super().__init__(mu=mu, filter_order=n, block_size=block_size)

        self.algorithm = "APA"
        self.eps = 1e-8

    def update_step(
        self, e_n: NDArray[np.float64], x_n: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Update for APA: mu * X[n]^T(X[n]X[n]^T + dI)^-1 @ e[n]

        Args:
            e_n (NDArray[np.float64]): Error vec (n)
            x_n (NDArray[np.float64]): vector[n] of array X, the noise estimate

        Returns:
            NDArray[np.float64]: The weight update vector for APA.
        """
        # getting the identity
        identity = np.eye(self.block_size)
        # calculating part inside of parens for legibility
        inner = x_n.T @ x_n + self.eps * identity
        return self.mu * x_n @ (np.linalg.inv(inner) @ e_n)
