import numpy as np
from numpy.typing import NDArray

from adaptive_filter.filter_models.filter_model import FilterModel


# Given that the only thing specific to LMS is its update step ->
# We only update the update step
class LMS(FilterModel):
    # updating the update step for LMS algorithm
    def update_step(self, e_n: float, x_n: NDArray[np.float64]) -> NDArray[np.float64]:
        # returning en * Xn * mu to be added with self.Wn to get updated W
        return self.mu * e_n * x_n
