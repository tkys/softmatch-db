"""Tests for core/zipfian.py."""

from __future__ import annotations

import numpy as np
import pytest

from softmatch_db.core.zipfian import compute_zipfian_norms


class TestZipfianNorms:
    """Unit tests for Zipfian whitening norm computation."""

    def test_uniform_frequency_equal_norms(self) -> None:
        """With uniform frequencies, all norms should be approximately equal."""
        rng = np.random.default_rng(123)
        v, d = 50, 8
        emb = rng.standard_normal((v, d)).astype(np.float32)
        freq = np.ones(v, dtype=np.int64) * 100

        norms = compute_zipfian_norms(emb, freq)
        # All norms should be close to each other.
        assert norms.shape == (v,)
        relative_spread = (norms.max() - norms.min()) / (norms.mean() + 1e-12)
        # With truly uniform freq, spread should be moderate.
        assert relative_spread < 5.0

    def test_zipf_frequency_high_freq_smaller_norm(self) -> None:
        """High-frequency tokens should have smaller norms than rare tokens."""
        rng = np.random.default_rng(456)
        v, d = 100, 16
        emb = rng.standard_normal((v, d)).astype(np.float32)
        # Zipf distribution: token 0 is most frequent.
        freq = np.array([10000 // (i + 1) for i in range(v)], dtype=np.int64)
        freq[freq == 0] = 1

        norms = compute_zipfian_norms(emb, freq)

        # Average norm of top-10 should be less than average of bottom-10.
        top10_mean = norms[:10].mean()
        bot10_mean = norms[-10:].mean()
        assert top10_mean < bot10_mean

    def test_zero_frequency_gets_default(self) -> None:
        """Tokens with zero frequency should get the default large norm."""
        v, d = 10, 4
        emb = np.eye(v, d, dtype=np.float32)
        freq = np.zeros(v, dtype=np.int64)
        freq[0] = 100
        freq[1] = 50

        norms = compute_zipfian_norms(emb, freq)
        # Tokens 2..9 have freq=0 → norm should be 1e10.
        for i in range(2, v):
            assert norms[i] == pytest.approx(1e10)

    def test_all_zero_frequency(self) -> None:
        """All zero frequencies → all default norms."""
        v, d = 5, 3
        emb = np.ones((v, d), dtype=np.float32)
        freq = np.zeros(v, dtype=np.int64)

        norms = compute_zipfian_norms(emb, freq)
        assert np.all(norms == pytest.approx(1e10))

    def test_output_shape_and_dtype(self) -> None:
        """Output should be float32 with correct shape."""
        v, d = 20, 8
        rng = np.random.default_rng(789)
        emb = rng.standard_normal((v, d)).astype(np.float32)
        freq = rng.integers(1, 1000, size=v).astype(np.int64)

        norms = compute_zipfian_norms(emb, freq)
        assert norms.shape == (v,)
        assert norms.dtype == np.float32
