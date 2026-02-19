"""Tests for core/sorted_index.py."""

from __future__ import annotations

import numpy as np
import pytest

from softmatch_db.core.sorted_index import (
    SortedIndex,
    compress,
    compress_build,
    seq_hash,
)


class TestCompress:
    """Tests for the compress function."""

    def test_empty_lower(self) -> None:
        """Empty sequence in lower-bound mode → all zeros."""
        assert compress([], 0) == (0, 0, 0, 0)

    def test_empty_upper(self) -> None:
        """Empty sequence in upper-bound mode → max u64 in h0."""
        h0, h1, h2, h3 = compress([], 1)
        assert h0 == 0xFFFFFFFFFFFFFFFF

    def test_upper_increments_last(self) -> None:
        """Upper-bound mode should increment the last token by 1."""
        lo = compress([5, 10], 0)
        hi = compress([5, 10], 1)
        # hi should be strictly greater than lo.
        assert hi > lo

    def test_sort_order_preserved(self) -> None:
        """Lexicographic order of token sequences should be preserved."""
        seqs = [
            [1, 2, 3],
            [1, 2, 4],
            [1, 3, 0],
            [2, 0, 0],
        ]
        hashes = [compress(s, 0) for s in seqs]
        for i in range(len(hashes) - 1):
            assert hashes[i] < hashes[i + 1], f"Order violated at {i}"

    def test_roundtrip_with_build(self) -> None:
        """compress and compress_build should produce same h0-h3."""
        tokens = list(range(12))
        arr = np.array(tokens, dtype=np.uint32)
        build_result = compress_build(arr, 0)
        query_result = compress(tokens, 0)
        assert build_result[:4] == query_result


class TestSeqHash:
    """Tests for the seq_hash function."""

    def test_deterministic(self) -> None:
        """Same input → same hash."""
        s = [1, 2, 3]
        assert seq_hash(s) == seq_hash(s)

    def test_different_seqs_different_hashes(self) -> None:
        """Different sequences should (almost certainly) have different hashes."""
        h1 = seq_hash([1, 2, 3])
        h2 = seq_hash([1, 2, 4])
        assert h1 != h2


class TestSortedIndex:
    """Tests for the SortedIndex class."""

    def test_build_and_exists(self, synthetic_corpus: np.ndarray) -> None:
        """Subsequences present in corpus should be found."""
        idx = SortedIndex.build(synthetic_corpus)
        # The first 3 tokens of the corpus should exist.
        first3 = [int(synthetic_corpus[i]) for i in range(3)]
        assert idx.exists(first3) is True

    def test_nonexistent_sequence(self) -> None:
        """A sequence not in corpus should not be found."""
        corpus = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.uint32)
        idx = SortedIndex.build(corpus)
        assert idx.exists([99, 98, 97]) is False

    def test_count(self) -> None:
        """Count should reflect actual occurrences."""
        # Corpus with repeated pattern.
        tokens = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 99],
            dtype=np.uint32,
        )
        idx = SortedIndex.build(tokens)
        # [1,2,3] appears at position 0 and 12.
        assert idx.count([1, 2, 3]) == 2

    def test_empty_corpus(self) -> None:
        """Empty corpus should always return False/0."""
        idx = SortedIndex.build(np.array([], dtype=np.uint32))
        assert idx.exists([1]) is False
        assert idx.count([1]) == 0

    def test_caching(self) -> None:
        """Repeated queries should use cache."""
        corpus = np.array(list(range(20)), dtype=np.uint32)
        idx = SortedIndex.build(corpus)
        seq = [0, 1, 2]
        _ = idx.exists(seq)
        assert seq_hash(seq) in idx.cache
        # Second call should still work.
        assert idx.exists(seq) is True
