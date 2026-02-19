#!/usr/bin/env python
"""Japanese Wikipedia コーパスの準備スクリプト。

HuggingFace datasets から前処理済み日本語Wikipediaを取得し、
1文1行のテキストファイルに変換して保存する。

使い方:
    uv run python bench/prepare_corpus.py --output data/wiki_ja.txt --max-sentences 100000
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def clean_line(text: str) -> str:
    """不要な記号・空白を除去して1行に正規化する。"""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def extract_sentences(text: str) -> list[str]:
    """記事テキストを文単位に分割する。

    句点（。！？）で分割し、最低10文字以上の行を有効とする。
    """
    # 段落分割
    paras = re.split(r"\n+", text)
    sentences: list[str] = []
    for para in paras:
        para = clean_line(para)
        if not para:
            continue
        # 句点での分割
        parts = re.split(r"(?<=[。！？])", para)
        for p in parts:
            p = p.strip()
            if len(p) >= 10:
                sentences.append(p)
    return sentences


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HuggingFace Wikipedia (ja) を1文1行コーパスに変換"
    )
    parser.add_argument(
        "--output", default="data/wiki_ja.txt",
        help="出力ファイルパス (default: data/wiki_ja.txt)"
    )
    parser.add_argument(
        "--max-sentences", type=int, default=100_000,
        help="最大行数 (default: 100000)"
    )
    parser.add_argument(
        "--max-articles", type=int, default=10_000,
        help="最大記事数 (default: 10000)"
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("datasets ライブラリでWikipedia (ja) をダウンロード中...")
    print("(初回は数百MB のダウンロードが発生します)")

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets が必要です。 uv sync --group dev を実行してください。")
        sys.exit(1)

    # wikimedia/wikipedia の新API形式を使用
    # streaming=True で全部DLせず順次取得
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.ja",
        split="train",
        streaming=True,
    )

    total_sentences = 0
    total_articles = 0
    print(f"出力先: {output_path}")
    print(f"目標: {args.max_sentences:,} 文 / {args.max_articles:,} 記事上限")

    with open(output_path, "w", encoding="utf-8") as fout:
        for article in ds:
            if total_articles >= args.max_articles:
                break
            text = article.get("text", "")
            if not text:
                continue
            sents = extract_sentences(text)
            for s in sents:
                if total_sentences >= args.max_sentences:
                    break
                fout.write(s + "\n")
                total_sentences += 1
            total_articles += 1
            if total_articles % 500 == 0:
                print(
                    f"  {total_articles:,} 記事 / {total_sentences:,} 文 処理済み...",
                    end="\r",
                )

    print(f"\n完了: {total_articles:,} 記事 / {total_sentences:,} 文 → {output_path}")


if __name__ == "__main__":
    main()
