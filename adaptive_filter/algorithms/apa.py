import numpy as np
from numpy.typing import NDArray

from adaptive_filter.filter_models.filter_model import FilterModel


class APA(FilterModel):
    def __init__(self, mu: float, n: int) -> None:
        self.mu = mu
        self.N = n
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
        identity = x_n.shape[1]
        # calculating part inside of parens for legibility
        inner = x_n.T @ x_n + self.eps * identity
        return self.mu * x_n @ (np.linalg.inv(inner) @ e_n)
