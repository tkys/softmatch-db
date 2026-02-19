"""Embedding protocol definition."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

NDArrayF32 = NDArray[np.float32]


@runtime_checkable
class Embedding(Protocol):
    """Protocol that all embedding implementations must satisfy."""

    @property
    def embeddings(self) -> NDArrayF32:
        """Full embedding matrix of shape ``(V, D)``."""
        ...

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        ...

    def __call__(self, token_ids: list[int]) -> NDArrayF32:
        """Look up embeddings for a list of token ids.

        Args:
            token_ids: List of integer token ids.

        Returns:
            Array of shape ``(len(token_ids), dim)``.
        """
        ...
