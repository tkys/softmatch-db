"""FastText embedding loader.

Supports loading pre-computed embeddings from local ``.npy`` files.
HuggingFace Hub download is a future extension.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

NDArrayF32 = NDArray[np.float32]


class FastTextEmbedding:
    """FastText embedding wrapper backed by a NumPy array.

    Attributes:
        _embeddings: Normalised embedding matrix of shape ``(V, D)``.
    """

    def __init__(self, embeddings: NDArrayF32) -> None:
        self._embeddings = embeddings

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, model_name_or_path: str) -> FastTextEmbedding:
        """Load embeddings from a local ``.npy`` file.

        The file should contain a float32 array of shape ``(V, D)``.
        Embeddings are L2-normalised upon loading.

        Args:
            model_name_or_path: Path to a ``.npy`` file.

        Returns:
            ``FastTextEmbedding`` instance with normalised vectors.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        path = Path(model_name_or_path)
        if not path.exists():
            raise FileNotFoundError(f"Embedding file not found: {path}")

        raw = np.load(str(path)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normalised = (raw / norms).astype(np.float32)
        return cls(normalised)

    @classmethod
    def from_array(cls, embeddings: NDArrayF32) -> FastTextEmbedding:
        """Create from an in-memory array (for testing).

        Args:
            embeddings: Pre-normalised embedding matrix.

        Returns:
            ``FastTextEmbedding`` instance.
        """
        return cls(np.asarray(embeddings, dtype=np.float32))

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    @property
    def embeddings(self) -> NDArrayF32:
        """Full embedding matrix ``(V, D)``."""
        return self._embeddings

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return int(self._embeddings.shape[1])

    def __call__(self, token_ids: list[int]) -> NDArrayF32:
        """Look up embeddings by token id.

        Args:
            token_ids: List of integer ids.

        Returns:
            Array of shape ``(len(token_ids), dim)``.
        """
        return self._embeddings[token_ids]
