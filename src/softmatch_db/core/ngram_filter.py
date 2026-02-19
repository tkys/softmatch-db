"""N-gram bit-array filter for fast corpus-existence pruning.

Port of:
  - rust/src/search/z_bsearch.rs:328-375  (search_2, search_3)
  - rust/src/search/z_check.rs            (check_valid)

The filter stores a compact bitset recording which 2-gram and 3-gram
token pairs actually appear in the corpus. During beam search, candidates
containing unseen n-grams are pruned in O(1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

NDArrayU64 = NDArray[np.uint64]
NDArrayU32 = NDArray[np.uint32]


@dataclass
class NgramBitFilter:
    """Compact bitset for 2-gram and 3-gram existence checks.

    Attributes:
        pair_bits: Bit array for 2-gram existence, packed as uint64.
        pair_cons: Number of top-frequency tokens covered by pair filter.
        trio_bits: Bit array for 3-gram existence, packed as uint64.
        trio_cons: Number of top-frequency tokens covered by trio filter.
    """

    pair_bits: NDArrayU64
    pair_cons: int
    trio_bits: NDArrayU64
    trio_cons: int

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def has_pair(self, u1: int, u2: int) -> bool:
        """Check whether 2-gram ``[u1, u2]`` exists in the corpus.

        Args:
            u1: First token id.
            u2: Second token id.

        Returns:
            True if the pair is known to exist **or** if either token id
            falls outside the tracked range (conservative: never prune
            what we cannot confirm).
        """
        if u1 >= self.pair_cons or u2 >= self.pair_cons:
            return True  # Unknown → don't prune.
        h = u1 * self.pair_cons + u2
        return bool((int(self.pair_bits[h >> 6]) >> (h & 63)) & 1)

    def has_trio(self, u1: int, u2: int, u3: int) -> bool:
        """Check whether 3-gram ``[u1, u2, u3]`` exists in the corpus.

        Uses the combinatorial hash from z_bsearch.rs:362-364.

        Args:
            u1: First token id.
            u2: Second token id.
            u3: Third token id.

        Returns:
            True if the trio is known to exist or if the ids are out of range.
        """
        if u1 + u2 + u3 >= self.trio_cons:
            return True  # Unknown → don't prune.
        t = self.trio_cons
        h = (
            (t * (t + 1) * (t + 2)) // 6
            - ((t - u1) * (t + 1 - u1) * (t + 2 - u1)) // 6
            + u2 * (t - u1)
            - u2 * (u2 - 1) // 2
            + u3
        )
        return bool((int(self.trio_bits[h >> 6]) >> (h & 63)) & 1)

    def check_valid(self, seq: Sequence[int]) -> bool:
        """Validate the tail of *seq* against pair and trio filters.

        Mirrors ``z_check.rs::check_valid``: returns False when the
        sequence is *obviously* absent from the corpus.

        Args:
            seq: Token id sequence being constructed.

        Returns:
            False if the tail n-grams are confirmed absent; True otherwise.
        """
        n = len(seq)
        if n >= 2 and not self.has_pair(seq[n - 2], seq[n - 1]):
            return False
        if n >= 3 and not self.has_trio(seq[n - 3], seq[n - 2], seq[n - 1]):
            return False
        return True

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build_from_corpus(
        cls,
        tokens: NDArrayU32,
        pair_cons: int,
        trio_cons: int,
    ) -> NgramBitFilter:
        """Scan a token array and build pair/trio bitsets.

        Uses NumPy vectorized operations to avoid Python loops.

        Args:
            tokens: Flat 1-D array of corpus token ids.
            pair_cons: Track 2-grams for token ids in ``[0, pair_cons)``.
            trio_cons: Track 3-grams for token ids in ``[0, trio_cons)``.

        Returns:
            Populated ``NgramBitFilter``.
        """
        pair_size = (pair_cons * pair_cons + 63) // 64
        trio_total = (trio_cons * (trio_cons + 1) * (trio_cons + 2)) // 6
        trio_size = (trio_total + 63) // 64

        pair_bits = np.zeros(pair_size, dtype=np.uint64)
        trio_bits = np.zeros(trio_size, dtype=np.uint64)

        n = len(tokens)
        tok = tokens.astype(np.int64)

        # --- 2-grams (vectorized) ---
        if n >= 2:
            u1_all = tok[:-1]
            u2_all = tok[1:]
            mask = (u1_all < pair_cons) & (u2_all < pair_cons)
            u1_f = u1_all[mask]
            u2_f = u2_all[mask]
            h_pair = u1_f * pair_cons + u2_f

            # Deduplicate to avoid redundant bit-setting
            h_unique = np.unique(h_pair)
            word_idx = (h_unique >> 6).astype(np.intp)
            bit_pos = (h_unique & 63).astype(np.uint64)
            for w, b in zip(word_idx, bit_pos):
                pair_bits[w] |= np.uint64(1) << b

        # --- 3-grams (vectorized) ---
        if n >= 3:
            u1_all = tok[:-2]
            u2_all = tok[1:-1]
            u3_all = tok[2:]
            mask = (u1_all + u2_all + u3_all) < trio_cons
            u1_f = u1_all[mask]
            u2_f = u2_all[mask]
            u3_f = u3_all[mask]

            t = np.int64(trio_cons)
            h_trio = (
                (t * (t + 1) * (t + 2)) // 6
                - ((t - u1_f) * (t + 1 - u1_f) * (t + 2 - u1_f)) // 6
                + u2_f * (t - u1_f)
                - u2_f * (u2_f - 1) // 2
                + u3_f
            )
            h_unique = np.unique(h_trio)
            word_idx = (h_unique >> 6).astype(np.intp)
            bit_pos = (h_unique & 63).astype(np.uint64)
            for w, b in zip(word_idx, bit_pos):
                trio_bits[w] |= np.uint64(1) << b

        return cls(
            pair_bits=pair_bits,
            pair_cons=pair_cons,
            trio_bits=trio_bits,
            trio_cons=trio_cons,
        )

    # ------------------------------------------------------------------
    # Serialisation (for DuckDB BLOB storage)
    # ------------------------------------------------------------------

    def to_bytes(self) -> tuple[bytes, bytes]:
        """Serialise pair and trio bitsets to raw bytes.

        Returns:
            Tuple of ``(pair_blob, trio_blob)``.
        """
        return self.pair_bits.tobytes(), self.trio_bits.tobytes()

    @classmethod
    def from_bytes(
        cls,
        pair_cons: int,
        trio_cons: int,
        pair_blob: bytes,
        trio_blob: bytes,
    ) -> NgramBitFilter:
        """Deserialise from raw bytes.

        Args:
            pair_cons: Pair filter coverage size.
            trio_cons: Trio filter coverage size.
            pair_blob: Raw bytes of pair bitset.
            trio_blob: Raw bytes of trio bitset.

        Returns:
            Restored ``NgramBitFilter``.
        """
        pair_bits = np.frombuffer(pair_blob, dtype=np.uint64).copy()
        trio_bits = np.frombuffer(trio_blob, dtype=np.uint64).copy()
        return cls(
            pair_bits=pair_bits,
            pair_cons=pair_cons,
            trio_bits=trio_bits,
            trio_cons=trio_cons,
        )
