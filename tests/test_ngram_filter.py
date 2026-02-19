"""Tests for core/ngram_filter.py."""

from __future__ import annotations

import numpy as np
import pytest

from softmatch_db.core.ngram_filter import NgramBitFilter


class TestNgramBitFilter:
    """Unit tests for the N-gram bit filter."""

    def test_build_and_has_pair(self) -> None:
        """Pairs present in corpus should be found."""
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
        nf = NgramBitFilter.build_from_corpus(tokens, pair_cons=10, trio_cons=10)
        assert nf.has_pair(1, 2) is True
        assert nf.has_pair(2, 3) is True
        assert nf.has_pair(4, 5) is True
        # Pair not in corpus.
        assert nf.has_pair(1, 3) is False
        assert nf.has_pair(5, 1) is False

    def test_build_and_has_trio(self) -> None:
        """Trios present in corpus should be found."""
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
        nf = NgramBitFilter.build_from_corpus(tokens, pair_cons=10, trio_cons=10)
        assert nf.has_trio(1, 2, 3) is True
        assert nf.has_trio(3, 4, 5) is True
        # Trio not in corpus.
        assert nf.has_trio(1, 3, 5) is False

    def test_out_of_range_returns_true(self) -> None:
        """Token ids beyond cons should return True (conservative)."""
        tokens = np.array([1, 2, 3], dtype=np.uint32)
        nf = NgramBitFilter.build_from_corpus(tokens, pair_cons=3, trio_cons=3)
        # u1=5 >= pair_cons=3 → unknown → True.
        assert nf.has_pair(5, 1) is True
        # u1+u2+u3=6 >= trio_cons=3 → unknown → True.
        assert nf.has_trio(2, 2, 2) is True

    def test_check_valid(self) -> None:
        """check_valid should reject sequences with invalid tail n-grams."""
        tokens = np.array([0, 1, 2, 3], dtype=np.uint32)
        nf = NgramBitFilter.build_from_corpus(tokens, pair_cons=10, trio_cons=10)
        assert nf.check_valid([0, 1]) is True
        assert nf.check_valid([0, 1, 2]) is True
        # Invalid pair at tail.
        assert nf.check_valid([0, 3, 1]) is False

    def test_serialisation_roundtrip(self) -> None:
        """to_bytes / from_bytes should preserve filter state."""
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
        nf = NgramBitFilter.build_from_corpus(tokens, pair_cons=10, trio_cons=10)
        pair_blob, trio_blob = nf.to_bytes()

        nf2 = NgramBitFilter.from_bytes(
            pair_cons=10, trio_cons=10,
            pair_blob=pair_blob, trio_blob=trio_blob,
        )
        # Check a few known pairs/trios.
        assert nf2.has_pair(1, 2) is True
        assert nf2.has_pair(5, 1) is False
        assert nf2.has_trio(2, 3, 4) is True
        assert nf2.has_trio(1, 3, 5) is False

    def test_empty_corpus(self) -> None:
        """Empty corpus should produce a filter where nothing is found."""
        tokens = np.array([], dtype=np.uint32)
        nf = NgramBitFilter.build_from_corpus(tokens, pair_cons=10, trio_cons=10)
        assert nf.has_pair(0, 1) is False
