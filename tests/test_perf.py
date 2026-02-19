"""pytest-benchmark によるパフォーマンステスト。

構築系: 合成データで計測（埋め込みに依存しない）。
検索系: 人工的に高スコアを注入した score_matrix で負荷を再現。

実行:
    uv run pytest tests/test_perf.py -v --benchmark-columns=mean,stddev,rounds
"""

from __future__ import annotations

import math

import duckdb
import numpy as np
import pytest

from softmatch_db.core.beam_search import beam_search
from softmatch_db.core.ngram_filter import NgramBitFilter
from softmatch_db.core.softmin import softmin, softmin_vec
from softmatch_db.core.sorted_index import SortedIndex
from softmatch_db.core.zipfian import compute_zipfian_norms


# ======================================================================
# 定数
# ======================================================================

VOCAB_SIZE = 5_000
EMBED_DIM = 300


# ======================================================================
# Session-scoped fixtures
# ======================================================================

@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def large_emb(rng: np.random.Generator) -> np.ndarray:
    raw = rng.standard_normal((VOCAB_SIZE, EMBED_DIM)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return (raw / np.maximum(norms, 1e-12)).astype(np.float32)


@pytest.fixture(scope="session")
def zipf_freq() -> np.ndarray:
    ranks = np.arange(1, VOCAB_SIZE + 1, dtype=np.float64)
    freq = (100_000 / ranks).astype(np.int64)
    freq[freq == 0] = 1
    return freq


# ======================================================================
# softmin
# ======================================================================

class TestSoftminPerf:

    def test_scalar(self, benchmark) -> None:
        benchmark(softmin, 0.7, 0.8)

    def test_vec_10k(self, benchmark) -> None:
        arr = np.full(10_000, 0.8, dtype=np.float32)
        benchmark(softmin_vec, 0.7, arr)


# ======================================================================
# Zipfian
# ======================================================================

class TestZipfianPerf:

    def test_5k_vocab(self, benchmark, large_emb, zipf_freq) -> None:
        benchmark(compute_zipfian_norms, large_emb, zipf_freq)


# ======================================================================
# N-gram filter: build (スケーリング)
# ======================================================================

class TestNgramFilterBuildPerf:

    @pytest.mark.parametrize("n_tokens", [10_000, 50_000, 100_000])
    def test_build(self, benchmark, n_tokens: int, rng: np.random.Generator) -> None:
        corpus = rng.integers(0, VOCAB_SIZE, size=n_tokens, dtype=np.uint32)
        benchmark(
            NgramBitFilter.build_from_corpus, corpus, 2000, 200
        )


# ======================================================================
# N-gram filter: lookup
# ======================================================================

class TestNgramFilterLookupPerf:

    @pytest.fixture(scope="class")
    def ngf(self) -> NgramBitFilter:
        rng = np.random.default_rng(0)
        corpus = rng.integers(0, VOCAB_SIZE, size=50_000, dtype=np.uint32)
        return NgramBitFilter.build_from_corpus(corpus, 2000, 200)

    def test_has_pair(self, benchmark, ngf) -> None:
        benchmark(ngf.has_pair, 10, 20)

    def test_has_trio(self, benchmark, ngf) -> None:
        benchmark(ngf.has_trio, 5, 10, 15)

    def test_check_valid(self, benchmark, ngf) -> None:
        benchmark(ngf.check_valid, [10, 20, 30, 40])


# ======================================================================
# SortedIndex: build (スケーリング)
# ======================================================================

class TestSortedIndexBuildPerf:

    @pytest.mark.parametrize("n_tokens", [5_000, 20_000, 50_000])
    def test_build(self, benchmark, n_tokens: int) -> None:
        rng = np.random.default_rng(42)
        corpus = rng.integers(0, VOCAB_SIZE, size=n_tokens, dtype=np.uint32)
        benchmark(SortedIndex.build, corpus)


# ======================================================================
# SortedIndex: lookup
# ======================================================================

class TestSortedIndexLookupPerf:

    @pytest.fixture(scope="class")
    def sidx(self) -> SortedIndex:
        rng = np.random.default_rng(0)
        corpus = rng.integers(0, VOCAB_SIZE, size=50_000, dtype=np.uint32)
        return SortedIndex.build(corpus)

    def test_exists_no_cache(self, benchmark, sidx: SortedIndex) -> None:
        def _f():
            sidx.cache.clear()
            return sidx.exists([10, 20, 30])
        benchmark(_f)

    def test_exists_cached(self, benchmark, sidx: SortedIndex) -> None:
        sidx.exists([10, 20, 30])  # warm cache
        benchmark(sidx.exists, [10, 20, 30])

    def test_count(self, benchmark, sidx: SortedIndex) -> None:
        def _f():
            sidx.cache.clear()
            return sidx.count([10, 20])
        benchmark(_f)


# ======================================================================
# Beam search（合成 score_matrix で本来の負荷を再現）
# ======================================================================

class TestBeamSearchPerf:
    """ビームサーチの実質的なパフォーマンスを測定する。

    ランダム埋め込みでは score_matrix が全て≈0 でビームが即座に死ぬため、
    人工的に高スコアのエントリを注入して、実際の検索ワークロードを模擬する。
    """

    @pytest.fixture(scope="class")
    def search_fixtures(self):
        """合成コーパス + 高スコア score_matrix を構築する。"""
        rng = np.random.default_rng(99)
        vocab = 500
        n_tokens = 10_000
        corpus = rng.integers(0, vocab, size=n_tokens, dtype=np.uint32)

        ngf = NgramBitFilter.build_from_corpus(corpus, pair_cons=vocab, trio_cons=vocab)
        sidx = SortedIndex.build(corpus)

        # Zipfian norms
        ranks = np.arange(1, vocab + 1, dtype=np.float64)
        freq = (10_000 / ranks).astype(np.int64)
        freq[freq == 0] = 1
        emb = rng.standard_normal((vocab, 32)).astype(np.float32)
        norms_emb = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms_emb, 1e-12)
        norm_sq = compute_zipfian_norms(emb, freq)

        # パターン: 3トークン
        pattern = [int(corpus[100]), int(corpus[101]), int(corpus[102])]

        # Score matrix: pattern[i] と同じ ID は 1.0、
        # コーパス内の近傍 ID には 0.5-0.8 のスコアを設定
        score_matrix = np.full((3, vocab), 0.0, dtype=np.float32)
        for i, pid in enumerate(pattern):
            score_matrix[i, pid] = 1.0
            # コーパスに実在する n-gram の先頭トークンに高スコア
            for offset in range(50):
                pos = 100 + offset
                if pos < n_tokens:
                    t = int(corpus[pos])
                    if t < vocab:
                        score_matrix[i, t] = max(
                            score_matrix[i, t],
                            0.8 - 0.01 * offset
                        )

        return pattern, score_matrix, norm_sq, ngf, sidx

    def test_beam_search_3tok(self, benchmark, search_fixtures) -> None:
        pattern, score_matrix, norm_sq, ngf, sidx = search_fixtures

        def _search():
            sidx.cache.clear()
            return beam_search(
                pattern_tokens=pattern,
                score_matrix=score_matrix,
                norm_sq=norm_sq,
                ngram_filter=ngf,
                sorted_index=sidx,
                top_k=10,
                min_similarity=0.45,
                max_runtime=5.0,
            )

        results = benchmark.pedantic(_search, rounds=5, warmup_rounds=1)
        assert isinstance(results, list)
