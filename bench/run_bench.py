#!/usr/bin/env python
"""SoftMatch-DB ベンチマークスクリプト（E2E版）。

FastText 埋め込み + Wikipedia 日本語コーパスでの本格ベンチマーク。
各フェーズの所要時間・メモリ・検索の内部プロファイルを出力する。

前提:
    bench/prepare_corpus.py でコーパスを準備済み
    bench/prepare_fasttext.py で埋め込みを準備済み

使い方:
    # フル E2E（FastText + Wikipedia）
    uv run python bench/run_bench.py \
        --corpus data/wiki_ja.txt \
        --embedding data/fasttext_ja/embeddings.npy \
        --vocab data/fasttext_ja/vocab.json

    # コーパスサイズを絞って試す
    uv run python bench/run_bench.py \
        --corpus data/wiki_ja.txt \
        --embedding data/fasttext_ja/embeddings.npy \
        --vocab data/fasttext_ja/vocab.json \
        --max-lines 5000

    # ランダム埋め込みフォールバック（構築系ベンチのみ有意義）
    uv run python bench/run_bench.py \
        --corpus data/wiki_ja.txt \
        --max-lines 5000
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import tracemalloc
from pathlib import Path


class Timer:
    """コンテキストマネージャ形式のタイマー。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed = time.perf_counter() - self._start


def load_corpus_lines(corpus_path: str, max_lines: int | None) -> list[str]:
    """テキストファイルから行を読み込む。"""
    lines: list[str] = []
    with open(corpus_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def run_benchmark(args: argparse.Namespace) -> None:
    """メインベンチマーク実行。"""
    import duckdb
    import numpy as np

    import softmatch_db
    from softmatch_db.embeddings.fasttext_emb import FastTextEmbedding
    from softmatch_db.tokenizers.sudachi_tok import SudachiTokenizer

    results: dict = {"config": {}, "phases": {}, "search": []}
    has_real_emb = args.embedding and Path(args.embedding).exists()

    # ==================================================================
    # Phase 1: コーパス読み込み
    # ==================================================================
    print(f"\n{'='*70}")
    print("Phase 1: コーパス読み込み")
    with Timer("corpus_load") as t:
        lines = load_corpus_lines(args.corpus, args.max_lines)
    n_lines = len(lines)
    n_chars = sum(len(l) for l in lines)
    print(f"  {n_lines:,} 行, {n_chars:,} 文字  ({t.elapsed:.3f}s)")
    results["phases"]["corpus_load_s"] = t.elapsed
    results["config"]["n_lines"] = n_lines
    results["config"]["n_chars"] = n_chars

    # 一時コーパスファイル作成
    import tempfile
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    tmp.write("\n".join(lines))
    tmp.close()
    corpus_file = tmp.name

    # ==================================================================
    # Phase 2: SudachiPy トークナイザ構築
    # ==================================================================
    print(f"\nPhase 2: SudachiPy トークナイザ構築 (max_vocab={args.vocab_cap:,})")
    with Timer("tokenizer") as t:
        tokenizer = SudachiTokenizer.build_from_corpus(
            corpus_file, max_vocab=args.vocab_cap
        )
    vocab_size = tokenizer.vocab_size
    print(f"  語彙サイズ: {vocab_size:,}  ({t.elapsed:.3f}s)")
    results["phases"]["tokenizer_build_s"] = t.elapsed
    results["config"]["vocab_size"] = vocab_size

    # ==================================================================
    # Phase 3: 埋め込み準備
    # ==================================================================
    print(f"\nPhase 3: 埋め込み準備")
    with Timer("embedding") as t:
        if has_real_emb:
            raw = np.load(args.embedding).astype(np.float32)
            # FastText 語彙と SudachiPy 語彙のアライメント
            if args.vocab:
                emb, aligned_count = _align_embeddings(
                    raw, args.vocab, tokenizer, vocab_size
                )
                print(f"  FastText ロード: {raw.shape} → アライメント後: {emb.embeddings.shape}")
                print(f"  語彙カバレッジ: {aligned_count:,}/{vocab_size:,} "
                      f"({100*aligned_count/vocab_size:.1f}%)")
                results["config"]["vocab_coverage"] = aligned_count
            else:
                # vocab.json なし: 先頭 vocab_size 行をそのまま使う
                emb = FastTextEmbedding.from_array(raw[:vocab_size])
                print(f"  FastText ロード: {emb.embeddings.shape} (直接マップ)")
            results["config"]["embed_type"] = "fasttext"
        else:
            rng = np.random.default_rng(42)
            raw = rng.standard_normal((vocab_size, args.embed_dim)).astype(np.float32)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            emb = FastTextEmbedding.from_array(raw / np.maximum(norms, 1e-12))
            print(f"  ⚠ ランダム埋め込み: {emb.embeddings.shape}")
            print(f"    (検索ベンチは意味的に無効 — 構築系のみ有効)")
            results["config"]["embed_type"] = "random"
    results["phases"]["embedding_load_s"] = t.elapsed
    results["config"]["embed_dim"] = emb.dim

    # ==================================================================
    # Phase 4: インデックス構築（フェーズ別計測）
    # ==================================================================
    print(f"\nPhase 4: インデックス構築")
    print(f"  pair_cons={args.pair_cons}, trio_cons={args.trio_cons}")
    con = duckdb.connect()

    tracemalloc.start()
    with Timer("full_build") as t_build:
        softmatch_db.register(con)
        softmatch_db.build(
            con=con,
            corpus_path=corpus_file,
            lang="ja",
            tokenizer=tokenizer,
            embedding=emb,
            pair_cons=args.pair_cons,
            trio_cons=args.trio_cons,
        )
    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  構築完了: {t_build.elapsed:.3f}s")
    print(f"  メモリ: current={mem_current/1e6:.1f}MB  peak={mem_peak/1e6:.1f}MB")
    results["phases"]["index_build_s"] = t_build.elapsed
    results["phases"]["mem_peak_mb"] = mem_peak / 1e6

    # サイズ情報
    from softmatch_db.duckdb_ext import get_searcher
    searcher = get_searcher()
    if searcher:
        n_idx = len(searcher.sorted_index.hashes)
        print(f"  SortedIndex エントリ: {n_idx:,}")
        results["config"]["sorted_index_entries"] = n_idx

    vocab_count = con.sql("SELECT COUNT(*) FROM softmatch_vocab").fetchone()[0]
    corpus_count = con.sql("SELECT COUNT(*) FROM softmatch_corpus").fetchone()[0]
    total_tokens = con.sql(
        "SELECT SUM(len(tokens)) FROM softmatch_corpus"
    ).fetchone()[0]
    print(f"  vocab: {vocab_count:,}, corpus: {corpus_count:,} 文, "
          f"total_tokens: {total_tokens:,}")
    results["config"]["total_tokens"] = total_tokens

    # ==================================================================
    # Phase 5: 検索ベンチマーク
    # ==================================================================
    queries = args.queries
    n_runs = 3
    print(f"\nPhase 5: 検索ベンチマーク ({len(queries)} クエリ × {n_runs} 回)")

    if not has_real_emb:
        print("  ⚠ ランダム埋め込みのため検索結果は意味的に無効")

    print(f"  {'クエリ':20s} | {'best':>8s} {'mean':>8s} {'結果数':>6s} | "
          f"{'sim_mat':>8s} {'beam':>8s} {'decode':>8s}")
    print(f"  {'-'*20}-+-{'-'*8}-{'-'*8}-{'-'*6}-+-{'-'*8}-{'-'*8}-{'-'*8}")

    for query in queries:
        times_ms: list[float] = []
        n_results: int = 0
        profile_sim: list[float] = []
        profile_beam: list[float] = []
        profile_decode: list[float] = []

        for _run in range(n_runs):
            gc.collect()
            if searcher:
                searcher.sorted_index.cache.clear()

            t0 = time.perf_counter()

            # 内部プロファイル: tokenize + similarity
            t_sim_start = time.perf_counter()
            if searcher:
                ids = searcher._tokenize_and_encode(query)
                score_matrix = searcher._compute_similarity(ids) if ids else None
            t_sim_end = time.perf_counter()

            # 内部プロファイル: beam search
            t_beam_start = time.perf_counter()
            if searcher and ids and score_matrix is not None:
                from softmatch_db.core.beam_search import beam_search
                raw_results = beam_search(
                    pattern_tokens=ids,
                    score_matrix=score_matrix,
                    norm_sq=searcher.norm_sq,
                    ngram_filter=searcher.ngram_filter,
                    sorted_index=searcher.sorted_index,
                    top_k=args.top_k,
                    min_similarity=args.threshold,
                    max_runtime=args.max_runtime,
                )
            else:
                raw_results = []
            t_beam_end = time.perf_counter()

            # 内部プロファイル: decode
            t_dec_start = time.perf_counter()
            if searcher:
                df = searcher.decode_results(raw_results)
            t_dec_end = time.perf_counter()

            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)
            n_results = len(raw_results)
            profile_sim.append((t_sim_end - t_sim_start) * 1000)
            profile_beam.append((t_beam_end - t_beam_start) * 1000)
            profile_decode.append((t_dec_end - t_dec_start) * 1000)

        best = min(times_ms)
        mean = sum(times_ms) / len(times_ms)
        sim_mean = sum(profile_sim) / len(profile_sim)
        beam_mean = sum(profile_beam) / len(profile_beam)
        dec_mean = sum(profile_decode) / len(profile_decode)

        q_display = query[:18]
        print(
            f"  {q_display:20s} | {best:7.1f}ms {mean:7.1f}ms {n_results:5d}  | "
            f"{sim_mean:7.1f}ms {beam_mean:7.1f}ms {dec_mean:7.1f}ms"
        )

        results["search"].append({
            "query": query,
            "best_ms": best,
            "mean_ms": mean,
            "n_results": n_results,
            "sim_matrix_ms": sim_mean,
            "beam_search_ms": beam_mean,
            "decode_ms": dec_mean,
        })

    # ==================================================================
    # サマリ
    # ==================================================================
    print(f"\n{'='*70}")
    print("サマリ")
    print(f"  コーパス          : {n_lines:,} 行 / {total_tokens:,} トークン")
    print(f"  語彙サイズ        : {vocab_size:,}")
    print(f"  埋め込み          : {results['config']['embed_type']} ({emb.dim}d)")
    print(f"  インデックス構築  : {t_build.elapsed:.3f}s")
    print(f"  ピークメモリ      : {mem_peak/1e6:.1f} MB")
    if results["search"]:
        avg_ms = sum(r["best_ms"] for r in results["search"]) / len(results["search"])
        avg_beam = sum(r["beam_search_ms"] for r in results["search"]) / len(results["search"])
        print(f"  検索 (best) 平均  : {avg_ms:.1f}ms (beam: {avg_beam:.1f}ms)")

    # JSON 出力
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f"\n結果を {args.output_json} に保存しました。")

    Path(corpus_file).unlink(missing_ok=True)
    con.close()


def _align_embeddings(
    fasttext_matrix,
    vocab_json_path: str,
    tokenizer,
    vocab_size: int,
):
    """FastText 語彙と SudachiPy 語彙をアライメントする。

    FastText の vocab.json を読み込み、SudachiPy のトークナイザ語彙と
    マッチするエントリを埋め込み行列にマッピングする。
    マッチしない語彙はランダムベクトルで埋める。
    """
    import numpy as np

    from softmatch_db.embeddings.fasttext_emb import FastTextEmbedding

    with open(vocab_json_path, encoding="utf-8") as fh:
        ft_vocab: dict[str, int] = json.load(fh)

    dim = fasttext_matrix.shape[1]
    aligned = np.zeros((vocab_size, dim), dtype=np.float32)

    # ランダムで初期化（未マッチ語彙用）
    rng = np.random.default_rng(0)
    for i in range(vocab_size):
        v = rng.standard_normal(dim).astype(np.float32)
        aligned[i] = v / max(np.linalg.norm(v), 1e-12)

    # マッチング
    aligned_count = 0
    if hasattr(tokenizer, "_id_to_token"):
        for tid, tok in tokenizer._id_to_token.items():
            if tid < vocab_size and tok in ft_vocab:
                ft_idx = ft_vocab[tok]
                if ft_idx < len(fasttext_matrix):
                    aligned[tid] = fasttext_matrix[ft_idx]
                    aligned_count += 1

    return FastTextEmbedding.from_array(aligned), aligned_count


def main() -> None:
    parser = argparse.ArgumentParser(description="SoftMatch-DB ベンチマーク")
    parser.add_argument("--corpus", required=True, help="1行1文コーパス")
    parser.add_argument("--embedding", default=None, help="FastText .npy")
    parser.add_argument("--vocab", default=None, help="FastText vocab.json")
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--pair-cons", type=int, default=2000)
    parser.add_argument("--trio-cons", type=int, default=200)
    parser.add_argument("--embed-dim", type=int, default=300)
    parser.add_argument("--vocab-cap", type=int, default=50_000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--max-runtime", type=float, default=10.0)
    parser.add_argument(
        "--queries", nargs="+",
        default=["金メダリスト", "オリンピック選手", "東京大学", "人工知能", "新幹線"],
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    if not Path(args.corpus).exists():
        print(f"ERROR: {args.corpus} が見つかりません。")
        print("  uv run python bench/prepare_corpus.py を実行してください。")
        sys.exit(1)

    run_benchmark(args)


if __name__ == "__main__":
    main()
