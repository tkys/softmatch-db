#!/usr/bin/env python
"""FastText 日本語埋め込みを取得し .npy に変換するスクリプト。

Facebook の cc.ja.300.bin (約4.2GB) をダウンロードし、
語彙と埋め込み行列を .npy ファイルに保存する。

前提:
    fasttext パッケージが必要。

使い方:
    uv run python bench/prepare_fasttext.py \
        --output-dir data/fasttext_ja \
        --max-vocab 50000
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np


def download_fasttext_bin(output_dir: Path) -> Path:
    """FastText cc.ja.300.bin.gz をダウンロードして解凍する。

    Args:
        output_dir: 保存先ディレクトリ。

    Returns:
        解凍済み .bin ファイルのパス。
    """
    bin_path = output_dir / "cc.ja.300.bin"
    gz_path = output_dir / "cc.ja.300.bin.gz"

    if bin_path.exists():
        print(f"  既に存在: {bin_path}")
        return bin_path

    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.bin.gz"

    if not gz_path.exists():
        print(f"  ダウンロード中: {url}")
        print("  (約4.2GB — 回線速度により10分〜1時間)")
        import urllib.request

        def _report(block_num: int, block_size: int, total_size: int) -> None:
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                mb = downloaded / 1e6
                print(f"\r  {mb:.0f} MB ({pct}%)", end="", flush=True)

        urllib.request.urlretrieve(url, str(gz_path), reporthook=_report)
        print()
    else:
        print(f"  gz 既に存在: {gz_path}")

    # 解凍
    print(f"  解凍中: {gz_path} → {bin_path}")
    with gzip.open(str(gz_path), "rb") as f_in, open(str(bin_path), "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return bin_path


def load_fasttext_bin(bin_path: Path, max_vocab: int) -> tuple[list[str], np.ndarray]:
    """FastText .bin ファイルを直接読んで語彙と行列を取り出す。

    fasttext パッケージを使わず、バイナリフォーマットを直接パースする。
    元実装と同様に .bin から .vec 相当のデータを取り出す。

    ただし .bin は複雑なため、まず .vec テキスト形式の取得を試み、
    失敗したら fasttext パッケージにフォールバックする。

    Args:
        bin_path: cc.ja.300.bin のパス。
        max_vocab: 取得する最大語彙数。

    Returns:
        (語彙リスト, 埋め込み行列 (V, 300))。
    """
    # 方法1: fasttext パッケージ
    try:
        import fasttext
        import fasttext.util

        print(f"  fasttext パッケージで読み込み中: {bin_path}")
        model = fasttext.load_model(str(bin_path))
        words = model.get_words()[:max_vocab]
        dim = model.get_dimension()
        matrix = np.zeros((len(words), dim), dtype=np.float32)
        for i, w in enumerate(words):
            matrix[i] = model.get_word_vector(w)
            if (i + 1) % 10000 == 0:
                print(f"\r  {i+1:,}/{len(words):,} 語読み込み済み", end="", flush=True)
        print()
        return words, matrix
    except ImportError:
        pass

    # 方法2: .vec テキストファイルを探す
    vec_path = bin_path.with_suffix(".vec")
    if not vec_path.exists():
        vec_gz_path = bin_path.parent / "cc.ja.300.vec.gz"
        if not vec_gz_path.exists():
            url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz"
            print(f"  .vec.gz ダウンロード中: {url}")
            print("  (約1.4GB)")
            import urllib.request

            def _report(block_num: int, block_size: int, total_size: int) -> None:
                downloaded = block_num * block_size
                if total_size > 0:
                    pct = min(100, downloaded * 100 // total_size)
                    mb = downloaded / 1e6
                    print(f"\r  {mb:.0f} MB ({pct}%)", end="", flush=True)

            urllib.request.urlretrieve(url, str(vec_gz_path), reporthook=_report)
            print()

        print(f"  解凍中: {vec_gz_path}")
        with gzip.open(str(vec_gz_path), "rt", encoding="utf-8") as f_in, \
             open(str(vec_path), "w", encoding="utf-8") as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"  .vec テキストから読み込み中: {vec_path}")
    words: list[str] = []
    vectors: list[np.ndarray] = []
    with open(str(vec_path), encoding="utf-8", errors="ignore") as fh:
        header = fh.readline().split()
        total_vocab = int(header[0])
        dim = int(header[1])
        print(f"  総語彙: {total_vocab:,}, 次元: {dim}")
        for line_no, line in enumerate(fh):
            if len(words) >= max_vocab:
                break
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            words.append(word)
            vectors.append(vec)
            if (line_no + 1) % 10000 == 0:
                print(f"\r  {len(words):,}/{max_vocab:,} 語読み込み済み", end="", flush=True)
    print()

    matrix = np.stack(vectors)
    return words, matrix


def download_vec_gz(output_dir: Path) -> Path:
    """FastText cc.ja.300.vec.gz を直接ダウンロードする（.bin より軽量）。

    Args:
        output_dir: 保存先ディレクトリ。

    Returns:
        .vec ファイルのパス。
    """
    vec_path = output_dir / "cc.ja.300.vec"
    vec_gz_path = output_dir / "cc.ja.300.vec.gz"

    if vec_path.exists():
        print(f"  既に存在: {vec_path}")
        return vec_path

    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz"
    if not vec_gz_path.exists():
        print(f"  ダウンロード中: {url}")
        print("  (約1.4GB)")
        import urllib.request

        def _report(block_num: int, block_size: int, total_size: int) -> None:
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                mb = downloaded / 1e6
                print(f"\r  {mb:.0f} MB ({pct}%)", end="", flush=True)

        urllib.request.urlretrieve(url, str(vec_gz_path), reporthook=_report)
        print()
    else:
        print(f"  gz 既に存在: {vec_gz_path}")

    print(f"  解凍中: {vec_gz_path}")
    with gzip.open(str(vec_gz_path), "rt", encoding="utf-8", errors="ignore") as f_in, \
         open(str(vec_path), "w", encoding="utf-8") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return vec_path


def load_vec_file(vec_path: Path, max_vocab: int) -> tuple[list[str], np.ndarray]:
    """FastText .vec テキストファイルから語彙と行列を読み込む。

    Args:
        vec_path: .vec ファイルのパス。
        max_vocab: 取得する最大語彙数。

    Returns:
        (語彙リスト, 埋め込み行列 (V, 300))。
    """
    words: list[str] = []
    vectors: list[np.ndarray] = []
    with open(str(vec_path), encoding="utf-8", errors="ignore") as fh:
        header = fh.readline().split()
        total_vocab = int(header[0])
        dim = int(header[1])
        print(f"  総語彙: {total_vocab:,}, 次元: {dim}")
        for line_no, line in enumerate(fh):
            if len(words) >= max_vocab:
                break
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            words.append(word)
            vectors.append(vec)
            if (line_no + 1) % 10000 == 0:
                print(f"\r  {len(words):,}/{max_vocab:,} 語読み込み済み", end="", flush=True)
    print()
    matrix = np.stack(vectors)
    return words, matrix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FastText 日本語埋め込みの取得と .npy 変換"
    )
    parser.add_argument(
        "--output-dir", default="data/fasttext_ja",
        help="出力ディレクトリ (default: data/fasttext_ja)"
    )
    parser.add_argument(
        "--max-vocab", type=int, default=50_000,
        help="最大語彙数 (default: 50000)"
    )
    parser.add_argument(
        "--use-bin", action="store_true",
        help=".bin 形式を使う（デフォルトは軽量な .vec 形式）"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    emb_path = output_dir / "embeddings.npy"
    vocab_path = output_dir / "vocab.json"

    if emb_path.exists() and vocab_path.exists():
        print(f"既に変換済み: {emb_path}, {vocab_path}")
        emb = np.load(str(emb_path))
        print(f"  形状: {emb.shape}")
        return

    print("=" * 60)
    if args.use_bin:
        # .bin 形式（4.2GB、fasttext パッケージ推奨）
        print("Step 1: FastText モデルの取得 (.bin)")
        download_fasttext_bin(output_dir)
        bin_path = output_dir / "cc.ja.300.bin"
        print("\nStep 2: 語彙・埋め込み行列の抽出")
        words, matrix = load_fasttext_bin(bin_path, args.max_vocab)
    else:
        # .vec 形式（1.4GB、軽量）
        print("Step 1: FastText ベクトルの取得 (.vec.gz)")
        vec_path = download_vec_gz(output_dir)
        print("\nStep 2: 語彙・埋め込み行列の抽出")
        words, matrix = load_vec_file(vec_path, args.max_vocab)

    print(f"  語彙: {len(words):,} 語, 行列: {matrix.shape}")

    # 正規化
    print("\nStep 3: L2 正規化")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    matrix_norm = (matrix / norms).astype(np.float32)

    # 保存
    print("\nStep 4: 保存")
    np.save(str(emb_path), matrix_norm)
    with open(str(vocab_path), "w", encoding="utf-8") as fh:
        json.dump(
            {w: i for i, w in enumerate(words)},
            fh, ensure_ascii=False, indent=2,
        )
    print(f"  埋め込み: {emb_path} ({emb_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  語彙: {vocab_path}")
    print("\n完了")


if __name__ == "__main__":
    main()
