"""SoftMin function — smooth approximation of min().

Port of: rust/src/helper.rs:153-157
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

NDArrayF32 = NDArray[np.float32]

# Constant controlling softmin sharpness.
# Larger ALPHA → closer to exact min().
ALPHA: float = 1.0e4
LOG2_ALPHA: float = math.log2(ALPHA)


def softmin(a: float, b: float) -> float:
    """Compute smooth minimum of two similarity scores.

    Both inputs and output are in [0, 1]. When both a and b are high,
    the result is high; when either is low, the result is pulled down
    (similar to a product but preserving [0, 1] scale).

    Args:
        a: First similarity score in [0, 1].
        b: Second similarity score in [0, 1].

    Returns:
        Merged similarity score in [0, 1].
    """
    s = ALPHA ** (1.0 - a) + ALPHA ** (1.0 - b) - 1.0
    return 1.0 - math.log2(s) / LOG2_ALPHA


def softmin_vec(a: float, b_array: NDArrayF32) -> NDArrayF32:
    """Vectorised softmin: scalar *a* against every element of *b_array*.

    Args:
        a: Scalar similarity score in [0, 1].
        b_array: 1-D array of similarity scores.

    Returns:
        Array of merged similarities, same shape as *b_array*.
    """
    s = ALPHA ** (1.0 - a) + np.power(ALPHA, 1.0 - b_array) - 1.0
    return (1.0 - np.log2(s) / LOG2_ALPHA).astype(np.float32)
