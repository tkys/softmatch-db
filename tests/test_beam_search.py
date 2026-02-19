"""Tests for core/beam_search.py."""

from __future__ import annotations

import numpy as np
import pytest

from softmatch_db.core.beam_search import (
    SearchResult,
    beam_search,
    check_subsequence,
    pareto_prune,
)
from softmatch_db.core.ngram_filter import NgramBitFilter
from softmatch_db.core.sorted_index import SortedIndex


class TestParetoprune:
    """Tests for Pareto dominance pruning."""

    def test_no_dominated(self) -> None:
        """No candidate is dominated → all kept."""
        cands = [
            (0.9, 0.5, [1], 0),
            (0.5, 0.9, [1], 0),
        ]
        result = pareto_prune(cands)
        assert len(result) == 2

    def test_dominated_removed(self) -> None:
        """A dominated candidate should be removed."""
        cands = [
            (0.9, 0.9, [1], 0),
            (0.5, 0.5, [1], 0),
        ]
        result = pareto_prune(cands)
        assert len(result) == 1
        assert result[0][0] == pytest.approx(0.9)

    def test_different_seqs_kept(self) -> None:
        """Candidates with different seqs are independent groups."""
        cands = [
            (0.9, 0.9, [1], 0),
            (0.5, 0.5, [2], 0),
        ]
        result = pareto_prune(cands)
        assert len(result) == 2


class TestCheckSubsequence:
    """Tests for contiguous subsequence check."""

    def test_present(self) -> None:
        assert check_subsequence([1, 2, 3, 4], [2, 3]) is True

    def test_absent(self) -> None:
        assert check_subsequence([1, 2, 3, 4], [2, 4]) is False

    def test_equal(self) -> None:
        assert check_subsequence([1, 2, 3], [1, 2, 3]) is True

    def test_b_longer(self) -> None:
        assert check_subsequence([1, 2], [1, 2, 3]) is False


class TestBeamSearch:
    """Integration-level tests for beam search."""

    def test_exact_match_returns_self(self) -> None:
        """Query that exactly matches a corpus segment should be found."""
        # Tiny corpus.
        corpus = np.array(
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
             10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 999],
            dtype=np.uint32,
        )
        vocab_size = 130
        pat_len = 3
        pattern = [10, 20, 30]

        # Build infrastructure.
        ngf = NgramBitFilter.build_from_corpus(corpus, pair_cons=vocab_size, trio_cons=vocab_size)
        sidx = SortedIndex.build(corpus)

        # Score matrix: identity-like (exact match gets 1.0).
        score_matrix = np.zeros((pat_len, vocab_size), dtype=np.float32)
        for i, tid in enumerate(pattern):
            score_matrix[i, tid] = 1.0
            # Add some noise for nearby tokens.
            for d in range(1, 5):
                if tid + d < vocab_size:
                    score_matrix[i, tid + d] = 0.3

        norm_sq = np.full(vocab_size, 100.0, dtype=np.float32)

        results = beam_search(
            pattern_tokens=pattern,
            score_matrix=score_matrix,
            norm_sq=norm_sq,
            ngram_filter=ngf,
            sorted_index=sidx,
            top_k=5,
            min_similarity=0.5,
            max_runtime=5.0,
        )

        # The exact pattern should be among results.
        found_exact = any(r.tokens == pattern for r in results)
        assert found_exact, f"Exact match not found. Results: {results}"

    def test_empty_pattern(self) -> None:
        """Empty pattern should return empty results."""
        corpus = np.array(list(range(20)), dtype=np.uint32)
        ngf = NgramBitFilter.build_from_corpus(corpus, pair_cons=20, trio_cons=20)
        sidx = SortedIndex.build(corpus)
        score_matrix = np.zeros((0, 20), dtype=np.float32)
        norm_sq = np.ones(20, dtype=np.float32)

        results = beam_search(
            pattern_tokens=[],
            score_matrix=score_matrix,
            norm_sq=norm_sq,
            ngram_filter=ngf,
            sorted_index=sidx,
        )
        assert results == []
