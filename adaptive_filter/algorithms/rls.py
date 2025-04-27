import numpy as np
from numpy.typing import NDArray

from adaptive_filter.filter_models.filter_model import FilterModel


class RLS(FilterModel):
    def __init__(self, mu: float, n: int) -> None:
        # writing to super
        super().__init__(mu=mu, filter_order=n)
        self.algorithm = "RLS"
        self.P = (1 / 1) * np.eye(n)

    # updating the update step for LMS algorithm
    def update_step(self, e_n: float, x_n: NDArray[np.float64]) -> NDArray[np.float64]:
        """Update for RLS: R_k = (x_n / (mu + x_n^Tx_n)

        Args:
            e_n (float): Error sample (n)
            x_n (NDArray[np.float64]): vector[n] of array X, the noise estimate

        Returns:
            NDArray[np.float64]: The weight update vector for RLS.
        """
        # calculate K_t
        K_t = self.P @ x_n / (self.mu + x_n.T @ (self.P @ x_n))

        # updating P_t
        self.P = (1 / self.mu) * (
            self.P
            - (
                (self.P @ (x_n[:, None] * x_n[None, :]) @ self.P)
                / (self.mu + x_n.T @ (self.P @ x_n))
            )
        )

        return K_t * e_n
