# SoftMatch-DB

**SoftMatcha2 の検索アルゴリズムを学習・再実装した Python プロジェクト + DuckDB + Web UI**

[SoftMatcha2](https://github.com/softmatcha/softmatcha2)（Rust 実装）の論文とソースコードを読み解き、そのコアとなる検索アルゴリズムを Python/NumPy で再実装したものです。公式の完全な再現ではなく、アルゴリズムの理解と実験を目的とした学習プロジェクトです。

## What is SoftMatcha2?

SoftMatcha2 は、テキストコーパス中から**意味的に類似するパターン**を高速に検索するアルゴリズムです。単語の完全一致ではなく、FastText 埋め込みによるコサイン類似度を用いた「ソフトマッチング」を行います。

例: `京都` で検索 → `京都`(1.00), `大阪`(0.70), `奈良`(0.66) のような意味的に関連するパターンがヒット

## Screenshots

### 検索結果（「京都」）
意味的に類似する地名がスコア付きで一覧表示されます。スコアバーは青(0.8+)/緑(0.6+)/黄(0.45+) の色段階。

![Search results for 京都](docs/screenshots/02_search_kyoto.png)

### KWIC 出現例（クリックで展開）
結果行をクリックすると、コーパス中の出現例がハイライト付きで表示されます。

![KWIC expanded view](docs/screenshots/03_kwic_expanded.png)

### マルチトークン検索（「人工知能」→「人工|知能」）
複合語は MeCab で分割され、マルチトークンのソフトマッチングが実行されます。

![Search for 人工知能 with KWIC](docs/screenshots/04_search_ai.png)

## 公式 SoftMatcha2 との関係

本プロジェクトは公式の完全な再現ではありません。以下に差異を明示します。

### 再現したもの（検索アルゴリズムの核）

| コンポーネント | 公式 (Rust) | 本プロジェクト (Python) |
|---|---|---|
| SoftMin スコアリング | `z_core.rs` | `core/softmin.py` — 同等 |
| N-gram ビットフィルタ | `z_check.rs` | `core/ngram_filter.py` — pair/trio bitset、同等 |
| Beam search 探索 | `z_enumerate.rs` | `core/beam_search.py` — match/delete/insert + Pareto 枝刈り |
| cand_next 先読み | `z_enumerate.rs:108-129` | `core/sorted_index.py` — 考え方を移植、データ構造は異なる |
| Zipfian whitening | 頻度加重正規化 | `core/zipfian.py` — 同等 |

### 対応していないもの・異なる設計

| 公式の特徴 | 本プロジェクトの設計 | 理由 |
|---|---|---|
| **ディスクベース mmap** — 1.4T トークン規模のインデックスをファイルから直接参照 | **全オンメモリ** — Python list/NumPy array で保持 | 学習目的のため単純化を優先。RAM が上限（~50M トークン） |
| **サフィックス配列** — ファイル上のバイナリフォーマット | **4×u64 圧縮ハッシュ** — Python bisect で二分探索 | mmap 前提のフォーマットを Python で扱うのが困難なため簡略化 |
| **Rust 高速 I/O** — ゼロコピー、SIMD 最適化 | **Pure Python + NumPy BLAS** | 可読性と実験のしやすさを優先 |
| **分散処理対応** — 大規模コーパスのシャーディング | **単一プロセス** | 学習規模では不要 |
| **インデックスファイルフォーマット** — 独自バイナリ | **DuckDB** — SQL でアクセス可能なストレージ | 実験・デバッグの利便性 |

### 規模の比較

| | 公式 | 本プロジェクト | 倍率 |
|---|---|---|---|
| トークン数 | 1.4 兆 | 120 万（目標: 5000 万） | 公式の 1/100 万 |
| 言語 | Rust | Python | — |
| インデックス | ファイル mmap | オンメモリ | — |

## Features

- **検索アルゴリズムの再実装**: softmin + n-gram フィルタ + sorted index + cand_next 先読みの3層フィルタリング
- **DuckDB 統合**: コーパス・語彙・インデックスを単一の `.duckdb` ファイルに格納、SQL でデバッグ可能
- **Web 検索 UI**: FastAPI + vanilla JS によるブラウザベースの検索インターフェース
  - KWIC（Keyword-In-Context）出現例表示
  - スコアバー色段階表示
  - トークン分割の可視化
- **MeCab トークナイザ**: 公式と同一の MeCab/ipadic（FastText cc.ja.300 の学習時トークナイザと一致）
- **80 テスト**: unit 64 + benchmark 16

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Web UI (index.html)                            │
│  ├── /api/search → soft-matching search         │
│  ├── /api/kwic   → corpus example lookup        │
│  └── /api/stats  → dataset statistics           │
├─────────────────────────────────────────────────┤
│  FastAPI (app.py)                               │
├─────────────────────────────────────────────────┤
│  softmatch_db Python API                        │
│  ├── register(con) / build(con, ...) / search() │
│  └── Searcher (in-memory, no SQL at query time) │
├─────────────────────────────────────────────────┤
│  Core Algorithm                                 │
│  ├── beam_search  (cand_next + Pareto pruning)  │
│  ├── sorted_index (4×u64 hash, bisect)          │
│  ├── ngram_filter (pair/trio bitset)             │
│  ├── softmin      (scoring function)            │
│  └── zipfian      (frequency whitening)         │
├─────────────────────────────────────────────────┤
│  DuckDB Storage                                 │
│  ├── softmatch_corpus (text + token arrays)     │
│  ├── softmatch_vocab  (embeddings + norms)      │
│  └── softmatch_ngram_index (bitset BLOBs)       │
└─────────────────────────────────────────────────┘
```

## Quick Start

### 前提条件

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (パッケージマネージャ)
- MeCab + ipadic（自動インストールされます）

### 1. セットアップ

```bash
git clone https://github.com/tkys/softmatch-db.git
cd softmatch-db
uv sync
```

### 2. データ準備

#### コーパス取得（Wikipedia 日本語）

```bash
uv run python bench/prepare_corpus.py \
    --output data/wiki_ja.txt \
    --max-sentences 50000
```

初回は HuggingFace から数百 MB のダウンロードが発生します。`--max-sentences` で規模を調整できます。

#### FastText 埋め込み取得

```bash
uv run python bench/prepare_fasttext.py \
    --output-dir data/fasttext_ja \
    --max-vocab 50000
```

`cc.ja.300.vec.gz`（約 1.4GB）をダウンロードし、`.npy` + `vocab.json` に変換します。

### 3. インデックス構築

```bash
uv run python web/build_index.py
```

`data/wiki_ja.txt` + `data/fasttext_ja/` から `web/index.duckdb` を生成します。50k 文で約 37 秒。

### 4. Web アプリ起動

```bash
uv run python -m web.app
```

ブラウザで http://localhost:8000 にアクセス。

### 5. テスト実行

```bash
uv run pytest tests/ -x -q
```

## Python API

```python
import duckdb
import softmatch_db

con = duckdb.connect("web/index.duckdb")
softmatch_db.register(con)

# ソフトマッチング検索
results = softmatch_db.search(con, "人工知能", top_k=20, threshold=0.45)
print(results)
#    rank       tokens      text    score  count
# 0     1  [694, 210]   人工 知能   1.0000     15
# 1     2  [4811, 210]  機械 知能   0.7234      3
# ...
```

## Project Structure

```
softmatch-db/
├── src/softmatch_db/
│   ├── __init__.py          # Public API: register, build, search
│   ├── core/
│   │   ├── beam_search.py   # Beam search with cand_next
│   │   ├── searcher.py      # Unified searcher + KWIC
│   │   ├── sorted_index.py  # 4×u64 hash index (bisect)
│   │   ├── ngram_filter.py  # Pair/trio bitset filter
│   │   ├── softmin.py       # SoftMin scoring
│   │   └── zipfian.py       # Zipfian whitening
│   ├── tokenizers/          # MeCab, SudachiPy, whitespace
│   ├── embeddings/          # FastText loader
│   ├── index/               # DuckDB schema + builder
│   └── duckdb_ext.py        # UDF registration
├── web/
│   ├── app.py               # FastAPI server
│   ├── build_index.py       # Index builder script
│   └── static/index.html    # Search UI
├── bench/
│   ├── prepare_corpus.py    # Wikipedia corpus downloader
│   ├── prepare_fasttext.py  # FastText embedding downloader
│   └── run_bench.py         # E2E benchmark
├── tests/                   # 80 tests (unit + benchmark)
├── docs/
│   └── SCALING.md           # Scaling guide & memory estimates
├── PROJECT_LOG.md           # Development log
└── pyproject.toml
```

## Performance

### 検索性能（50k 文 / 1.2M トークン）

| クエリ | レイテンシ | 結果例 |
|---|---|---|
| 京都 | 8 ms | 京都(1.00), 大阪(0.70), 奈良(0.66) |
| 人工知能 | 8 ms | 人工 知能(1.00), 機械 知能(0.72) |
| 新幹線 | 8 ms | 東海道 新幹線(0.76), 東北 新幹線(0.68) |

### ビルド性能

| コーパス規模 | トークン | ビルド時間 | DuckDB サイズ |
|---|---|---|---|
| 5,000文 | 147k | 33s | — |
| 50,000文 | 1.2M | 37s (最適化済み) | 61 MB |

### 最適化履歴

5段階の最適化で**ビルド 5.9x、検索 9.2x** 高速化を達成。詳細は [PROJECT_LOG.md](PROJECT_LOG.md) を参照。

## Scaling

大規模コーパスへのスケーリングガイドは [docs/SCALING.md](docs/SCALING.md) を参照。

| マシン | 推奨規模 | トークン | ビルド時間 |
|---|---|---|---|
| 16 GB RAM | 100万文 | 2400万 | 72 min |
| 64 GB RAM | 200万文 | 5000万 | 2.4 hr |

主な制約はオンメモリ設計（SortedIndex: 136 bytes/token）。ディスクベース mmap への移行で大幅なスケールアップが可能だが、本プロジェクトの範囲外。

## Docs

| ドキュメント | 内容 |
|---|---|
| [PROJECT_LOG.md](PROJECT_LOG.md) | 開発ログ（全セッションの達成事項・決定事項・ベンチマーク・公式との差異分析） |
| [docs/SCALING.md](docs/SCALING.md) | スケーリングガイド（メモリモデル・マシン別推奨・移行手順） |

## Acknowledgments

- [SoftMatcha2](https://github.com/softmatcha/softmatcha2) — Original Rust implementation by Tatsuya Hiraoka et al.
- [FastText](https://fasttext.cc/) — Pre-trained word embeddings (cc.ja.300)
- [DuckDB](https://duckdb.org/) — Embedded analytical database
- [MeCab](https://taku910.github.io/mecab/) + [ipadic](https://github.com/polm/ipadic-py) — Japanese morphological analyzer

## License

MIT
