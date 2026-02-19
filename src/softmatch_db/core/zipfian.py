"""Zipfian Whitening — frequency-aware embedding normalisation.

Port of: src/softmatcha/index/build.py:89-120

Common words get small norms (cheap to insert/delete),
content words get large norms (expensive).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

NDArrayF32 = NDArray[np.float32]
NDArrayI64 = NDArray[np.int64]


def compute_zipfian_norms(
    embeddings: NDArrayF32,
    frequencies: NDArrayI64,
    eps: float = 1e-11,
) -> NDArrayF32:
    """Compute Zipfian-whitened squared norms for each vocabulary token.

    Args:
        embeddings: Token embedding matrix of shape ``(V, D)``.
        frequencies: Token frequency counts of shape ``(V,)``.
        eps: Floor value for eigenvalues to avoid division by zero.

    Returns:
        Array of shape ``(V,)`` containing squared norms after whitening.
        Tokens with zero frequency receive a large default value (1e10).
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    frequencies = np.asarray(frequencies, dtype=np.float64)
    v_total, d = embeddings.shape
    norm_sq = np.full(v_total, 1e10, dtype=np.float32)

    # Only consider tokens that actually appear in the corpus.
    mask = frequencies > 0
    if not np.any(mask):
        return norm_sq

    emb_valid = embeddings[mask]
    freq_valid = frequencies[mask]

    # Frequency-weighted mean.
    p = freq_valid / freq_valid.sum()
    mu_hat = (p[:, None] * emb_valid).sum(axis=0)
    centered = emb_valid - mu_hat

    # Frequency-weighted covariance.
    sqrt_p = np.sqrt(p)
    w_p = centered * sqrt_p[:, None]
    covariance = w_p.T @ w_p

    # Eigen-decomposition and whitening matrix.
    eigvals, eigvecs = np.linalg.eigh(covariance)
    inv_s = np.diag(1.0 / np.sqrt(np.maximum(eigvals, eps)))
    # Note: build.py:115 overrides to eigvecs @ inv_S (not @ eigvecs.T).
    whitening_mat = eigvecs @ inv_s

    # Whitened embeddings and their norms.
    whitened_valid = centered @ whitening_mat
    norms = np.linalg.norm(whitened_valid, axis=1).astype(np.float32)

    # Store squared norms (matches build.py:128).
    norm_sq[mask] = norms ** 2
    return norm_sq
