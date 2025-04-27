from typing import Any

import numpy as np
from numpy.typing import NDArray

from adaptive_filter.filter_models.block_filter_model import FrequencyDomainAF


class FDLMS(FrequencyDomainAF):
    def __init__(self, mu: float, n: int, block_size: int) -> None:
        # initializing BlockFilterModel
        super().__init__(mu=mu, filter_order=n, block_size=block_size)
        self.algorithm = "FD_LMS"
        self.eps = 1e-6

    # updating the update step for LMS algorithm
    def update_step(
        self, e_f: NDArray[np.complex128], x_f: NDArray[np.complex128]
    ) -> NDArray[Any]:
        """Update for FDAF: FD LMS algorithm.

        Args:
            e_f (NDArray[np.complex128]): Block error in the frequency domain.
            x_f (NDArray[np.complex128]): Block noise estimate in the frequency domain.

        Returns:
            NDArray[np.float64]: The weight update vector for FDAF.
        """
        return self.mu * np.multiply(np.conj(x_f), e_f)
