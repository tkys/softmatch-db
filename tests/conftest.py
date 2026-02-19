"""Shared fixtures for SoftMatch-DB tests.

Provides small synthetic data (vocabulary, embeddings, tokens) so that
core unit tests do not require real NLP models.
"""

from __future__ import annotations

import numpy as np
import pytest

from softmatch_db.core.ngram_filter import NgramBitFilter
from softmatch_db.core.sorted_index import SortedIndex


# ======================================================================
# Small synthetic vocabulary & embeddings
# ======================================================================

VOCAB_SIZE = 100
EMBED_DIM = 16


@pytest.fixture()
def rng() -> np.random.Generator:
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture()
def synthetic_embeddings(rng: np.random.Generator) -> np.ndarray:
    """Random normalised embeddings of shape (VOCAB_SIZE, EMBED_DIM)."""
    raw = rng.standard_normal((VOCAB_SIZE, EMBED_DIM)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return (raw / np.maximum(norms, 1e-12)).astype(np.float32)


@pytest.fixture()
def synthetic_frequencies(rng: np.random.Generator) -> np.ndarray:
    """Zipf-like frequency distribution for VOCAB_SIZE tokens."""
    ranks = np.arange(1, VOCAB_SIZE + 1, dtype=np.float64)
    freq = (1000.0 / ranks).astype(np.int64)
    freq[freq == 0] = 1
    return freq


@pytest.fixture()
def synthetic_corpus(rng: np.random.Generator) -> np.ndarray:
    """Small synthetic corpus: 200 tokens drawn from [0, VOCAB_SIZE)."""
    return rng.integers(0, VOCAB_SIZE, size=200, dtype=np.uint32)


@pytest.fixture()
def ngram_filter(synthetic_corpus: np.ndarray) -> NgramBitFilter:
    """NgramBitFilter built from the synthetic corpus."""
    return NgramBitFilter.build_from_corpus(
        synthetic_corpus, pair_cons=VOCAB_SIZE, trio_cons=VOCAB_SIZE
    )


@pytest.fixture()
def sorted_index(synthetic_corpus: np.ndarray) -> SortedIndex:
    """SortedIndex built from the synthetic corpus."""
    return SortedIndex.build(synthetic_corpus)
