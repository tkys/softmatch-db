"""Simplified suffix array using sorted 4*u64 hashes.

Port of:
  - rust/src/helper.rs:15-62   (compress_build, compress)
  - rust/src/search/z_bsearch.rs:19-25  (get_hash)
  - rust/src/search/z_bsearch.rs:38-292 (binary search)

Each corpus position is hashed into a 4-tuple of uint64 values that
preserve lexicographic order of the underlying 12-token window. Binary
search on the sorted array gives O(log N) existence and count queries.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

NDArrayU32 = NDArray[np.uint32]
NDArrayU64 = NDArray[np.uint64]
NDArrayI64 = NDArray[np.int64]

_MASK20 = 0xFFFFF  # 20-bit mask (1048575)
_MASK64 = 0xFFFFFFFFFFFFFFFF


def _u64(v: int) -> int:
    """Ensure value fits in unsigned 64-bit range."""
    return v & _MASK64


def compress(seq: Sequence[int], mode: int = 0) -> tuple[int, int, int, int]:
    """Convert up to 12 tokens into a 4*u64 hash preserving sort order.

    Faithfully ports ``helper.rs:36-62``.

    Args:
        seq: Token id sequence (length <= 12).
        mode: 0 for lower bound, 1 for upper bound
              (last token incremented by 1).

    Returns:
        Tuple of four integers representing the compressed hash.
    """
    if len(seq) == 0:
        if mode == 1:
            return (_MASK64, 0, 0, 0)
        return (0, 0, 0, 0)

    token = [0] * 12
    for i, t in enumerate(seq):
        if i + 1 == len(seq) and mode == 1:
            token[i] = int(t) + 1
        else:
            token[i] = int(t)

    h0 = _u64(
        (token[0] << 44) | (token[1] << 24) | (token[2] << 4) | (token[3] >> 16)
    )
    h1 = _u64(
        ((token[3] & 0xFFFF) << 48) | (token[4] << 28) | (token[5] << 8) | (token[6] >> 12)
    )
    h2 = _u64(
        ((token[6] & 0xFFF) << 52) | (token[7] << 32) | (token[8] << 12) | (token[9] >> 8)
    )
    h3 = _u64(
        ((token[9] & 0xFF) << 56) | (token[10] << 36) | (token[11] << 16)
    )
    return (h0, h1, h2, h3)


def compress_build(tokens: Sequence[int], idx: int) -> tuple[int, int, int, int, int]:
    """Build-mode compression: 12 tokens -> 4*u64 hash + position.

    Ports ``helper.rs:15-23``.

    Args:
        tokens: Full corpus token array.
        idx: Starting position in the corpus.

    Returns:
        Tuple ``(h0, h1, h2, h3, idx)``.
    """
    t = [int(tokens[idx + i]) for i in range(12)]
    h0 = _u64((t[0] << 44) | (t[1] << 24) | (t[2] << 4) | (t[3] >> 16))
    h1 = _u64(((t[3] & 0xFFFF) << 48) | (t[4] << 28) | (t[5] << 8) | (t[6] >> 12))
    h2 = _u64(((t[6] & 0xFFF) << 52) | (t[7] << 32) | (t[8] << 12) | (t[9] >> 8))
    h3 = _u64(((t[9] & 0xFF) << 56) | (t[10] << 36) | (t[11] << 16))
    return (h0, h1, h2, h3, idx)


def seq_hash(seq: Sequence[int]) -> int:
    """Rolling hash for cache keys.  Ports z_bsearch.rs:19-25.

    Args:
        seq: Token id sequence.

    Returns:
        Hash value (u64 range).
    """
    h: int = 99999999
    for t in seq:
        h = _u64(8691201001 * h + (int(t) + 1234567890123))
    return h


def _build_hashes_vectorized(corpus_tokens: NDArrayU32) -> tuple[NDArrayU64, NDArrayU64, NDArrayU64, NDArrayU64]:
    """Vectorized hash computation for all 12-token windows.

    Computes h0, h1, h2, h3 for each position using NumPy array operations
    instead of a Python loop.

    Args:
        corpus_tokens: Flat uint32 array of corpus tokens.

    Returns:
        Four uint64 arrays (h0, h1, h2, h3) of length N-11.
    """
    n = len(corpus_tokens)
    # Build sliding window: shape (N-11, 12) — each row is 12 consecutive tokens
    # Use stride tricks for zero-copy view
    from numpy.lib.stride_tricks import as_strided
    itemsize = corpus_tokens.strides[0]
    windows = as_strided(
        corpus_tokens,
        shape=(n - 11, 12),
        strides=(itemsize, itemsize),
    ).astype(np.uint64)  # Cast to uint64 for bit operations

    t0, t1, t2, t3 = windows[:, 0], windows[:, 1], windows[:, 2], windows[:, 3]
    t4, t5, t6 = windows[:, 4], windows[:, 5], windows[:, 6]
    t7, t8, t9 = windows[:, 7], windows[:, 8], windows[:, 9]
    t10, t11 = windows[:, 10], windows[:, 11]

    h0 = (t0 << np.uint64(44)) | (t1 << np.uint64(24)) | (t2 << np.uint64(4)) | (t3 >> np.uint64(16))
    h1 = ((t3 & np.uint64(0xFFFF)) << np.uint64(48)) | (t4 << np.uint64(28)) | (t5 << np.uint64(8)) | (t6 >> np.uint64(12))
    h2 = ((t6 & np.uint64(0xFFF)) << np.uint64(52)) | (t7 << np.uint64(32)) | (t8 << np.uint64(12)) | (t9 >> np.uint64(8))
    h3 = ((t9 & np.uint64(0xFF)) << np.uint64(56)) | (t10 << np.uint64(36)) | (t11 << np.uint64(16))

    return h0, h1, h2, h3


@dataclass
class SortedIndex:
    """Sorted hash index for O(log N) n-gram existence and count queries.

    Attributes:
        hashes: List of 4-tuples ``(h0, h1, h2, h3)`` in sorted order.
        positions: Corpus position for each entry.
        cache: Memoisation dict ``{seq_hash: (exists, count)}``.
    """

    hashes: list[tuple[int, int, int, int]]
    positions: list[int]
    cache: dict[int, tuple[bool, int]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, corpus_tokens: NDArrayU32) -> SortedIndex:
        """Build sorted index from corpus token array.

        Uses NumPy vectorized hash computation and argsort for speed.

        Args:
            corpus_tokens: Flat 1-D uint32 array of all corpus tokens.

        Returns:
            Populated ``SortedIndex``.
        """
        n = len(corpus_tokens)
        if n < 12:
            return cls(hashes=[], positions=[])

        # Vectorized hash computation (NumPy)
        h0, h1, h2, h3 = _build_hashes_vectorized(corpus_tokens)
        num_entries = len(h0)

        # Lexicographic sort by (h0, h1, h2, h3) using structured array
        sort_keys = np.empty(num_entries, dtype=[
            ('h0', np.uint64), ('h1', np.uint64),
            ('h2', np.uint64), ('h3', np.uint64),
        ])
        sort_keys['h0'] = h0
        sort_keys['h1'] = h1
        sort_keys['h2'] = h2
        sort_keys['h3'] = h3

        order = np.argsort(sort_keys, order=('h0', 'h1', 'h2', 'h3'))

        # Apply sort order and convert to Python lists for bisect compatibility
        h0_sorted = h0[order]
        h1_sorted = h1[order]
        h2_sorted = h2[order]
        h3_sorted = h3[order]

        hashes = list(zip(
            h0_sorted.tolist(), h1_sorted.tolist(),
            h2_sorted.tolist(), h3_sorted.tolist(),
        ))
        positions = order.tolist()

        return cls(hashes=hashes, positions=positions)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def exists(self, seq: Sequence[int]) -> bool:
        """Check whether token sequence appears in the corpus.

        Args:
            seq: Token id sequence (length 1-12).

        Returns:
            True if at least one matching position exists.
        """
        if not seq or not self.hashes:
            return False

        h = seq_hash(seq)
        if h in self.cache:
            return self.cache[h][0]

        lo = compress(seq, 0)
        hi = compress(seq, 1)
        idx = bisect.bisect_left(self.hashes, lo)
        found = idx < len(self.hashes) and self.hashes[idx] < hi
        cnt = self.count_range(lo, hi) if found else 0
        self.cache[h] = (found, cnt)
        return found

    def count(self, seq: Sequence[int]) -> int:
        """Count occurrences of token sequence in the corpus.

        Args:
            seq: Token id sequence (length 1-12).

        Returns:
            Number of matching positions.
        """
        if not seq or not self.hashes:
            return 0

        h = seq_hash(seq)
        if h in self.cache:
            return self.cache[h][1]

        lo = compress(seq, 0)
        hi = compress(seq, 1)
        cnt = self.count_range(lo, hi)
        self.cache[h] = (cnt > 0, cnt)
        return cnt

    def count_range(
        self,
        lo: tuple[int, int, int, int],
        hi: tuple[int, int, int, int],
    ) -> int:
        """Count entries in ``[lo, hi)`` hash range.

        Args:
            lo: Lower bound hash (inclusive).
            hi: Upper bound hash (exclusive).

        Returns:
            Number of entries in range.
        """
        left = bisect.bisect_left(self.hashes, lo)
        right = bisect.bisect_left(self.hashes, hi)
        return right - left

    # ------------------------------------------------------------------
    # Lookahead (cand_next) — ports z_enumerate.rs:108-129
    # ------------------------------------------------------------------

    def get_start_pos(self, seq: Sequence[int]) -> int | None:
        """Find the lower-bound index in the sorted hash array for *seq*.

        Unlike ``exists()`` which only returns a bool, this returns the
        actual index so it can be used as ``start_pos`` for lookahead.

        Args:
            seq: Token id sequence (length 1-12).

        Returns:
            Index into ``self.hashes`` where *seq* first appears,
            or ``None`` if *seq* is not found.
        """
        if not seq or not self.hashes:
            return None
        lo = compress(seq, 0)
        hi = compress(seq, 1)
        idx = bisect.bisect_left(self.hashes, lo)
        if idx < len(self.hashes) and self.hashes[idx] < hi:
            return idx
        return None

    def get_next_tokens(
        self,
        seq: Sequence[int],
        start_pos: int,
        window: int = 50,
    ) -> list[tuple[int, int]]:
        """Extract tokens that follow *seq* in the corpus.

        Reads up to *window* entries from ``start_pos`` in the sorted
        hash array and extracts the token at position ``len(seq)``
        (i.e. the token immediately after the current prefix).

        Ports ``z_enumerate.rs:108-129``.

        Args:
            seq: Current token sequence (prefix).
            start_pos: Starting index in ``self.hashes``.
            window: Maximum number of entries to scan.

        Returns:
            List of ``(token_id, relative_position)`` tuples sorted
            by token id.  Empty if *seq* is too long or the range
            exceeds *window*.
        """
        seq_len = len(seq)
        if seq_len >= 12 or not self.hashes:
            return []

        hi = compress(seq, 1)
        target = min(len(self.hashes), start_pos + window)

        # Quick check: if range endpoint is still within bounds, use it.
        # If the target hash is still < hi, the range is wider than
        # our window → return empty to fall back to check_valid.
        if target < len(self.hashes) and self.hashes[target - 1] < hi:
            # More than `window` entries match → too frequent for lookahead.
            return []

        cand: list[tuple[int, int]] = []
        for i in range(start_pos, target):
            if self.hashes[i] >= hi:
                break
            token = retrieve_token_at(self.hashes[i], seq_len)
            cand.append((token, i - start_pos))

        cand.sort()
        return cand

    @staticmethod
    def _bsearch_token(
        cand_next: list[tuple[int, int]],
        token_id: int,
    ) -> int:
        """Binary search for *token_id* in a sorted ``cand_next`` list.

        Ports ``helper.rs:276-294``.

        Args:
            cand_next: Sorted list of ``(token_id, relative_pos)`` pairs.
            token_id: Token to search for.

        Returns:
            ``relative_pos`` if found, ``-1`` otherwise.
        """
        ng = -1
        ok = len(cand_next)
        while ok - ng > 1:
            mid = (ng + ok) // 2
            if cand_next[mid][0] >= token_id:
                ok = mid
            else:
                ng = mid
        if ok == len(cand_next) or cand_next[ok][0] != token_id:
            return -1
        return cand_next[ok][1]


def retrieve_token_at(
    hash_tuple: tuple[int, int, int, int],
    position: int,
) -> int:
    """Extract token id from a compressed 4*u64 hash at *position*.

    Faithfully ports ``helper.rs:75-115`` (``retrieve_value``).
    Each token occupies 20 bits within the packed hash.

    Args:
        hash_tuple: ``(h0, h1, h2, h3)`` compressed hash.
        position: Index in the 12-token window (0-11).

    Returns:
        Token id (20-bit value).

    Raises:
        ValueError: If *position* is outside ``[0, 11]``.
    """
    h0, h1, h2, h3 = hash_tuple
    if position == 0:
        return (h0 >> 44) & _MASK20
    if position == 1:
        return (h0 >> 24) & _MASK20
    if position == 2:
        return (h0 >> 4) & _MASK20
    if position == 3:
        return (((h0 & 0xF) << 16) | (h1 >> 48)) & _MASK20
    if position == 4:
        return (h1 >> 28) & _MASK20
    if position == 5:
        return (h1 >> 8) & _MASK20
    if position == 6:
        return (((h1 & 0xFF) << 12) | (h2 >> 52)) & _MASK20
    if position == 7:
        return (h2 >> 32) & _MASK20
    if position == 8:
        return (h2 >> 12) & _MASK20
    if position == 9:
        return (((h2 & 0xFFF) << 8) | (h3 >> 56)) & _MASK20
    if position == 10:
        return (h3 >> 36) & _MASK20
    if position == 11:
        return (h3 >> 16) & _MASK20
    raise ValueError(f"position must be 0-11, got {position}")
