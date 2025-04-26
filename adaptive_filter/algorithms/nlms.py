import numpy as np
from numpy.typing import NDArray

from adaptive_filter.filter_models.filter_model import FilterModel


class NLMS(FilterModel):
    def __init__(self, mu: float, n: int) -> None:
        self.mu = mu
        self.N = n
        self.algorithm = "NLMS"
        self.eps = 1e-8
        self.p = 0.0

    # updating the update step for LMS algorithm
    def update_step(self, e_n: float, x_n: NDArray[np.float64]) -> NDArray[np.float64]:
        """Update for NLMS: Calc. the normalizing step, key to NLMS algorithm. p(n) = eps + ||X(n)||**2

        Args:
            x_n (NDArray[np.float64]): vector[n] of array X, the noise estimate

        Returns:
            np.float64: The normalized step used in update step function.
        """
        normalized_step = self.eps + np.linalg.norm(x_n) ** 2
        return (self.mu / normalized_step) * e_n * x_n
