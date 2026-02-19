"""Tests for cand_next lookahead: retrieve_token_at, get_start_pos, get_next_tokens."""

from __future__ import annotations

import numpy as np
import pytest

from softmatch_db.core.sorted_index import (
    SortedIndex,
    compress,
    retrieve_token_at,
)


# ======================================================================
# retrieve_token_at
# ======================================================================


class TestRetrieveTokenAt:
    """Validate token extraction from compressed 4*u64 hashes."""

    @pytest.fixture()
    def known_tokens(self) -> list[int]:
        """12 distinct token IDs in the 20-bit range."""
        return [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]

    @pytest.fixture()
    def known_hash(self, known_tokens: list[int]) -> tuple[int, int, int, int]:
        """Compressed hash of the known tokens."""
        return compress(known_tokens, mode=0)

    def test_all_positions(
        self,
        known_tokens: list[int],
        known_hash: tuple[int, int, int, int],
    ) -> None:
        """Extracted token at each position matches the original."""
        for pos in range(12):
            extracted = retrieve_token_at(known_hash, pos)
            assert extracted == known_tokens[pos], (
                f"position {pos}: expected {known_tokens[pos]}, got {extracted}"
            )

    def test_boundary_values(self) -> None:
        """Tokens at the 20-bit boundary (0 and 0xFFFFF)."""
        tokens_zero = [0] * 12
        h = compress(tokens_zero, 0)
        for pos in range(12):
            assert retrieve_token_at(h, pos) == 0

        tokens_max = [0xFFFFF] * 12
        h = compress(tokens_max, 0)
        for pos in range(12):
            assert retrieve_token_at(h, pos) == 0xFFFFF

    def test_invalid_position_raises(self) -> None:
        """Position outside [0, 11] raises ValueError."""
        h = compress([1] * 12, 0)
        with pytest.raises(ValueError, match="position must be 0-11"):
            retrieve_token_at(h, 12)
        with pytest.raises(ValueError, match="position must be 0-11"):
            retrieve_token_at(h, -1)

    def test_single_nonzero_positions(self) -> None:
        """Only one position is nonzero; all others should be zero."""
        for target_pos in range(12):
            tokens = [0] * 12
            tokens[target_pos] = 42
            h = compress(tokens, 0)
            for pos in range(12):
                expected = 42 if pos == target_pos else 0
                actual = retrieve_token_at(h, pos)
                assert actual == expected, (
                    f"target={target_pos}, check pos={pos}: "
                    f"expected {expected}, got {actual}"
                )


# ======================================================================
# bsearch_token
# ======================================================================


class TestBsearchToken:
    """Validate binary search in cand_next lists."""

    def test_found(self) -> None:
        cand = [(10, 0), (20, 3), (30, 7)]
        assert SortedIndex._bsearch_token(cand, 20) == 3

    def test_not_found(self) -> None:
        cand = [(10, 0), (20, 3), (30, 7)]
        assert SortedIndex._bsearch_token(cand, 25) == -1

    def test_empty(self) -> None:
        assert SortedIndex._bsearch_token([], 5) == -1

    def test_first_element(self) -> None:
        cand = [(5, 2), (10, 4)]
        assert SortedIndex._bsearch_token(cand, 5) == 2

    def test_last_element(self) -> None:
        cand = [(5, 2), (10, 4)]
        assert SortedIndex._bsearch_token(cand, 10) == 4

    def test_single_element_found(self) -> None:
        cand = [(42, 9)]
        assert SortedIndex._bsearch_token(cand, 42) == 9

    def test_single_element_not_found(self) -> None:
        cand = [(42, 9)]
        assert SortedIndex._bsearch_token(cand, 41) == -1


# ======================================================================
# get_start_pos / get_next_tokens
# ======================================================================


@pytest.fixture()
def small_index() -> SortedIndex:
    """Build a SortedIndex from a small deterministic corpus.

    Corpus: tokens [10, 20, 30, 40, 50, 10, 20, 30, 40, 50, 10, 20, 30, 40, 50]
    This has at least 4 twelve-token windows (len=15, so 15-11=4 entries).
    """
    tokens = np.array(
        [10, 20, 30, 40, 50, 10, 20, 30, 40, 50, 10, 20, 30, 40, 50],
        dtype=np.uint32,
    )
    return SortedIndex.build(tokens)


class TestGetStartPos:
    """Validate get_start_pos returns correct index or None."""

    def test_existing_prefix(self, small_index: SortedIndex) -> None:
        """A prefix that exists should return a non-None index."""
        # Token 10 appears at multiple positions → prefix [10] should exist.
        pos = small_index.get_start_pos([10])
        assert pos is not None
        assert 0 <= pos < len(small_index.hashes)

    def test_nonexistent_prefix(self, small_index: SortedIndex) -> None:
        """A prefix that does not exist should return None."""
        # Token 99999 never appears.
        assert small_index.get_start_pos([99999]) is None

    def test_empty_seq(self, small_index: SortedIndex) -> None:
        """Empty sequence returns None."""
        assert small_index.get_start_pos([]) is None

    def test_two_token_prefix(self, small_index: SortedIndex) -> None:
        """A two-token prefix that exists."""
        pos = small_index.get_start_pos([10, 20])
        assert pos is not None


class TestGetNextTokens:
    """Validate get_next_tokens returns correct lookahead candidates."""

    def test_returns_correct_tokens(self, small_index: SortedIndex) -> None:
        """Next tokens after [10] should include 20."""
        sp = small_index.get_start_pos([10])
        assert sp is not None
        cand = small_index.get_next_tokens([10], sp)
        token_ids = [t for t, _p in cand]
        assert 20 in token_ids

    def test_empty_for_nonexistent(self, small_index: SortedIndex) -> None:
        """Non-existent prefix: get_start_pos returns None, so no lookahead."""
        sp = small_index.get_start_pos([99999])
        assert sp is None
        # get_next_tokens should only be called with a valid start_pos.
        # If prefix doesn't exist, the caller falls back to check_valid.

    def test_sorted_by_token_id(self, small_index: SortedIndex) -> None:
        """Result should be sorted by token_id."""
        sp = small_index.get_start_pos([10])
        if sp is not None:
            cand = small_index.get_next_tokens([10], sp)
            token_ids = [t for t, _p in cand]
            assert token_ids == sorted(token_ids)

    def test_long_seq_returns_empty(self, small_index: SortedIndex) -> None:
        """Sequence of length >= 12 returns empty."""
        cand = small_index.get_next_tokens(list(range(12)), 0)
        assert cand == []
