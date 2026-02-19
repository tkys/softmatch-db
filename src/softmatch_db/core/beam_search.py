"""Beam search — the core soft-matching enumeration algorithm.

Port of:
  - rust/src/search/z_core.rs       (compute)
  - rust/src/search/z_enumerate.rs  (enumerate_algorithm)
  - rust/src/helper.rs:208-263      (get_upper_convex, check_subsequence)

This is a single-threaded Python version with suffix-array lookahead
(``cand_next``) ported from z_enumerate.rs:108-129.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from softmatch_db.core.ngram_filter import NgramBitFilter
from softmatch_db.core.softmin import softmin
from softmatch_db.core.sorted_index import SortedIndex

NDArrayF32 = NDArray[np.float32]

# Default threshold schedule (z_core.rs:54-65).
ALPHA_SCHEDULE: list[float] = [
    0.99,
    0.80, 0.70, 0.60, 0.58, 0.57,
    0.56, 0.55, 0.54, 0.53, 0.52,
    0.51, 0.50, 0.49, 0.48, 0.47,
    0.46, 0.45, 0.44, 0.43, 0.42,
    0.41, 0.40, 0.39, 0.38, 0.37,
    0.36, 0.35, 0.34, 0.33, 0.32,
    0.31, 0.30, 0.29, 0.28, 0.27,
    0.26, 0.25, 0.24, 0.23, 0.22,
    0.21, 0.20,
]


@dataclass
class SearchResult:
    """A single search result from beam search.

    Attributes:
        tokens: Token id sequence of the matched pattern.
        score: Final score (``sim * mult``).
        count: Number of occurrences in the corpus.
    """

    tokens: list[int]
    score: float
    count: int


# ======================================================================
# Helper: Pareto pruning
# ======================================================================

def pareto_prune(
    candidates: list[tuple[float, float, list[int], int]],
) -> list[tuple[float, float, list[int], int]]:
    """Remove Pareto-dominated candidates within each sequence group.

    Ports ``helper.rs:208-231`` (get_upper_convex).

    Two candidates share a group when their token sequences are identical.
    Within a group, candidate *i* is dominated if there exists *j* with
    ``sim_j >= sim_i`` **and** ``mult_j >= mult_i`` (and they differ).

    The fourth element (``start_pos``) is carried through but not used
    for grouping or dominance — this matches the official Rust behaviour.

    Args:
        candidates: List of ``(sim, mult, seq, start_pos)`` tuples.

    Returns:
        Filtered list with dominated entries removed.
    """
    if not candidates:
        return []

    # Group by sequence.  Keep the first start_pos encountered per group.
    groups: dict[tuple[int, ...], list[tuple[float, float, int]]] = {}
    for sim, mult, seq, start_pos in candidates:
        key = tuple(seq)
        groups.setdefault(key, []).append((sim, mult, start_pos))

    result: list[tuple[float, float, list[int], int]] = []
    for key, points in groups.items():
        sm_pairs = [(s, m) for s, m, _ in points]
        kept_set = set(_upper_convex(sm_pairs))
        seq = list(key)
        for sim, mult, sp in points:
            if (sim, mult) in kept_set:
                result.append((sim, mult, seq, sp))
                kept_set.discard((sim, mult))
    return result


def _upper_convex(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Keep only non-dominated (sim, mult) pairs."""
    kept: list[tuple[float, float]] = []
    for i, (si, mi) in enumerate(points):
        dominated = False
        for j, (sj, mj) in enumerate(points):
            if i == j:
                continue
            if si == sj and mi == mj:
                # Tie-break: keep the earlier one.
                if j < i:
                    dominated = True
                    break
            elif sj >= si and mj >= mi:
                dominated = True
                break
        if not dominated:
            kept.append((si, mi))
    return kept


# ======================================================================
# Helper: subsequence check
# ======================================================================

def check_subsequence(a: list[int], b: list[int]) -> bool:
    """Check whether *b* is a contiguous subsequence of *a*.

    Ports ``helper.rs:244-263``.

    Args:
        a: Longer sequence.
        b: Shorter sequence (potential subsequence).

    Returns:
        True if *b* appears contiguously within *a*.
    """
    if len(a) < len(b):
        return False
    for i in range(len(a) - len(b) + 1):
        if a[i : i + len(b)] == b:
            return True
    return False


# ======================================================================
# Main beam search
# ======================================================================

def beam_search(
    pattern_tokens: list[int],
    score_matrix: NDArrayF32,
    norm_sq: NDArrayF32,
    ngram_filter: NgramBitFilter,
    sorted_index: SortedIndex,
    top_k: int = 20,
    min_similarity: float = 0.45,
    max_runtime: float = 10.0,
) -> list[SearchResult]:
    """Run beam search to find soft-matching patterns.

    Args:
        pattern_tokens: Token ids of the query pattern.
        score_matrix: Cosine similarity matrix ``(pat_len, V)`` where
            ``score_matrix[i][v]`` is the similarity between query
            position *i* and vocabulary token *v*.
        norm_sq: Zipfian-whitened squared norms for each vocabulary
            token, shape ``(V,)``.
        ngram_filter: N-gram bitset filter for pruning.
        sorted_index: Sorted hash index for corpus existence checks.
        top_k: Maximum number of results to return.
        min_similarity: Minimum acceptable final score.
        max_runtime: Time budget in seconds.

    Returns:
        List of ``SearchResult`` sorted by descending score.
    """
    pat_len = len(pattern_tokens)
    vocab_size = score_matrix.shape[1]
    t_start = time.monotonic()

    # ------------------------------------------------------------------
    # 1. Pre-compute costs (z_core.rs:70-139)
    # ------------------------------------------------------------------

    # (a) match_sim: for each pattern position, sorted (score, token_id) descending
    #     Vectorized with NumPy instead of P*V Python loop.
    min_alpha = min(ALPHA_SCHEDULE[-1], min_similarity)
    match_sim: list[list[tuple[float, int]]] = []
    score_clamped = np.maximum(score_matrix, 0.0)
    for i in range(pat_len):
        row_scores = score_clamped[i]
        mask = row_scores >= min_alpha
        indices = np.nonzero(mask)[0]
        scores_filtered = row_scores[indices]
        order = np.argsort(-scores_filtered)
        match_sim.append(list(zip(
            scores_filtered[order].tolist(),
            indices[order].tolist(),
        )))

    # (b) insertion coefficient (vectorized)
    inst_rank = 50
    top_count = max(vocab_size // 50, 1)
    temp_norms = np.sort(norm_sq[:top_count].copy())
    qual_idx = inst_rank * inst_rank + 100
    qual = float(temp_norms[qual_idx]) if qual_idx < len(temp_norms) else 1.0e9

    # Tokens with norm <= qual are insertion candidates.
    inst_mask = norm_sq[:vocab_size] <= qual
    inst_indices = np.nonzero(inst_mask)[0]
    inst_norms = norm_sq[inst_indices]
    inst_order = np.argsort(inst_norms)
    inst_indices = inst_indices[inst_order]
    inst_norms = inst_norms[inst_order]

    actual_inst_rank = inst_rank
    for i in range(1, min(inst_rank, len(inst_norms))):
        if inst_norms[i] >= 1.0e10:
            actual_inst_rank = i - 1
            break
    actual_inst_rank = min(actual_inst_rank, max(len(inst_norms) - 1, 0))

    coefficient = float(inst_norms[actual_inst_rank]) * pat_len / 5.0 if len(inst_norms) > 0 else 1.0
    coefficient = max(coefficient, 1e-12)

    # inst_mult: [(cost, token_id)] sorted by norm ascending.
    inst_mult: list[tuple[float, int]] = []
    for idx_j in range(len(inst_norms)):
        norm_v = float(inst_norms[idx_j])
        mult = math.exp(-norm_v / coefficient)
        if mult < min_alpha:
            break
        inst_mult.append((mult, int(inst_indices[idx_j])))

    # delt_mult: deletion cost for each pattern position.
    delt_mult: list[float] = []
    for i in range(pat_len):
        pid = pattern_tokens[i]
        if pid < len(norm_sq):
            delt_mult.append(math.exp(-float(norm_sq[pid]) / coefficient))
        else:
            delt_mult.append(0.0)

    # ------------------------------------------------------------------
    # 2. Iterate over threshold schedule
    # ------------------------------------------------------------------
    return_cands: list[tuple[list[int], float]] = []
    last_alpha = 1.01

    for alpha_idx, thres in enumerate(ALPHA_SCHEDULE):
        if thres < min_similarity:
            break

        # Enumerate at this threshold.
        cands = _enumerate(
            pat_len=pat_len,
            match_sim=match_sim,
            inst_mult=inst_mult,
            delt_mult=delt_mult,
            ngram_filter=ngram_filter,
            sorted_index=sorted_index,
            thres=thres,
            t_start=t_start,
            max_runtime=max_runtime,
        )

        # Timeout sentinel.
        if cands is None:
            break

        return_cands = cands
        last_alpha = thres

        if len(return_cands) >= top_k:
            break

    # ------------------------------------------------------------------
    # 3. Post-processing: remove subsequences, count occurrences
    # ------------------------------------------------------------------
    # Sort by score descending.
    return_cands.sort(key=lambda x: -x[1])

    final: list[SearchResult] = []
    for seq, score in return_cands:
        if not seq:
            continue
        # Skip if this pattern is a supersequence of an already-kept one.
        skip = False
        for prev in final:
            if check_subsequence(seq, prev.tokens):
                skip = True
                break
        if skip:
            continue

        cnt = sorted_index.count(seq)
        final.append(SearchResult(tokens=seq, score=score, count=cnt))
        if len(final) >= top_k:
            break

    return final


# ======================================================================
# Core enumeration (simplified z_enumerate)
# ======================================================================

def _enumerate(
    *,
    pat_len: int,
    match_sim: list[list[tuple[float, int]]],
    inst_mult: list[tuple[float, int]],
    delt_mult: list[float],
    ngram_filter: NgramBitFilter,
    sorted_index: SortedIndex,
    thres: float,
    t_start: float,
    max_runtime: float,
) -> list[tuple[list[int], float]] | None:
    """Single-threshold enumeration step with suffix-array lookahead.

    Each candidate is a 4-tuple ``(sim, mult, seq, start_pos)`` where
    ``start_pos`` tracks the position in the sorted hash index for
    the ``cand_next`` lookahead optimisation.

    Returns:
        List of ``(seq, score)`` tuples, or ``None`` on timeout.
    """
    max_inst = 4
    _bsearch = SortedIndex._bsearch_token

    # Candidates: (sim, mult, seq, start_pos)
    current: list[tuple[float, float, list[int], int]] = [(1.0, 1.0, [], 0)]

    for idx in range(pat_len):
        if time.monotonic() - t_start > max_runtime and thres < 0.95:
            return None

        next_raw: list[tuple[float, float, list[int], int]] = []

        for sim, mult, seq, start_pos in current:
            # Build cand_next once per candidate.
            cand_next = sorted_index.get_next_tokens(seq, start_pos)

            # --- (a) MATCH ---
            if len(seq) < 12:
                for s_iv, v in match_sim[idx]:
                    new_sim = softmin(sim, s_iv)
                    if new_sim * mult < thres:
                        break
                    new_seq = seq + [v]

                    if cand_next:
                        # Lookahead available: only accept if token
                        # actually follows prefix in corpus.
                        rel = _bsearch(cand_next, v)
                        if rel != -1:
                            next_raw.append((
                                new_sim, mult, new_seq,
                                start_pos + rel,
                            ))
                    else:
                        # No lookahead (empty seq or too frequent):
                        # fall back to n-gram filter + existence check.
                        if ngram_filter.check_valid(new_seq):
                            sp = sorted_index.get_start_pos(new_seq)
                            if sp is not None:
                                next_raw.append((
                                    new_sim, mult, new_seq, sp,
                                ))

            # --- (b) DELETE ---
            new_mult_d = mult * delt_mult[idx]
            if sim * new_mult_d >= thres:
                next_raw.append((sim, new_mult_d, list(seq), start_pos))

            # --- (c) INSERT (up to max_inst times) ---
            if len(seq) >= 1 and len(seq) < 12:
                cur_seq = list(seq)
                cur_mult = mult
                cur_sp = start_pos
                for _ins_round in range(max_inst):
                    if len(cur_seq) >= 12:
                        break
                    ins_cand = sorted_index.get_next_tokens(cur_seq, cur_sp)
                    inserted_any = False
                    for cost_v, v in inst_mult:
                        if sim * cur_mult * cost_v < thres:
                            break
                        ins_seq = cur_seq + [v]

                        if ins_cand:
                            rel = _bsearch(ins_cand, v)
                            if rel != -1:
                                next_raw.append((
                                    sim, cur_mult * cost_v, ins_seq,
                                    cur_sp + rel,
                                ))
                                inserted_any = True
                        else:
                            if ngram_filter.check_valid(ins_seq):
                                sp = sorted_index.get_start_pos(ins_seq)
                                if sp is not None:
                                    next_raw.append((
                                        sim, cur_mult * cost_v, ins_seq,
                                        sp,
                                    ))
                                    inserted_any = True
                    if not inserted_any:
                        break
                    # Simplified: only one round of insertion expansion.
                    break

        # Pareto prune.
        current = pareto_prune(next_raw)

        if time.monotonic() - t_start > max_runtime and thres < 0.95:
            return None

    # Collect final results.
    results: list[tuple[list[int], float]] = []
    for sim, mult, seq, _sp in current:
        if seq:
            results.append((seq, sim * mult))
    return results
