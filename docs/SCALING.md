# SoftMatch-DB スケーリングガイド

## 現状ベースライン（2026-02-19 時点）

| 項目 | 値 |
|---|---|
| コーパス | Wikipedia 日本語 50,000文 |
| トークン数 | 1,218,909 (1.2M) |
| トークン/文 | 24.4 |
| 語彙数 | 50,000 |
| ビルド時間 | 37秒 (最適化済み) |
| ピーク RAM (ビルド時) | 861 MB |
| ランタイム RAM | ~500 MB |
| DuckDB ファイル | 61 MB |
| 検索レイテンシ | ~8 ms/query |

## メモリモデル

### SortedIndex（最大のメモリ消費源）

SortedIndex はコーパス全トークンの圧縮ハッシュをオンメモリ保持する。

- **データ構造**: `list[tuple[u64, u64, u64, u64]]` + `list[int]`
- **1トークンあたり**: **136 bytes**
  - ハッシュ 4-tuple: ~100 bytes (Python object overhead含む)
  - positions リスト要素: ~36 bytes

### 固定コスト（コーパス規模に依存しない）

| コンポーネント | サイズ |
|---|---|
| FastText 埋め込み (50k vocab × 300d) | 60 MB |
| 語彙テーブル | 50 MB |
| トークナイザ (MeCab) | 20 MB |
| **合計固定** | **~130 MB** |

### メモリ見積もり式

```
ビルド時ピーク RAM (MB) ≈ 130 + (136 × total_tokens / 1,048,576) × 2.5
ランタイム RAM (MB)     ≈ 130 + (136 × total_tokens / 1,048,576)
```

ビルド時は `np.argsort` の一時配列で約 2.5 倍のオーバーヘッドが発生する。

## スケーリング見積もりテーブル

| 文数 | トークン | ビルドRAM | ランタイムRAM | ビルド時間 | DuckDB | 検索ms |
|---|---|---|---|---|---|---|
| 50,000 (現状) | 1.2M | 0.9 GB | 0.5 GB | 3.6 min | 61 MB | ~8 |
| 100,000 | 2.4M | 1.0 GB | 0.6 GB | 7 min | 122 MB | ~9 |
| 500,000 | 12M | 3.1 GB | 1.7 GB | 36 min | 610 MB | ~12 |
| **1,000,000** | **24M** | **5.7 GB** | **3.4 GB** | **72 min** | **1.2 GB** | **~14** |
| **2,000,000** | **49M** | **10.8 GB** | **6.6 GB** | **2.4 hr** | **2.4 GB** | **~16** |
| 5,000,000 | 122M | 26 GB | 16 GB | 6 hr | 6.1 GB | ~19 |
| 10,000,000 | 244M | 52 GB | 32 GB | 12 hr | 12.2 GB | ~22 |

## マシン別推奨構成

### 16GB RAM マシン

- **推奨**: 1,000,000文 / 2400万トークン
- ビルド RAM 5.7 GB、ランタイム 3.4 GB
- worker 1 で安全に運用可能

### 64GB RAM マシン（目標環境）

- **推奨**: 2,000,000文 / 5000万トークン
- ビルド RAM 10.8 GB、ランタイム 6.6 GB
- 4 workers (26.4 GB) でも余裕あり
- 「Wikipedia 日本語 200万文」はプロジェクトとして見栄えが良い

## Web アプリの同時リクエスト負荷

### 制約: GIL

検索処理は全て CPU バウンド（NumPy matmul + beam search）。
GIL により async でも実質直列処理。並列化には **マルチプロセス** が必要。

### worker 数 × メモリ

各 worker が Searcher を独立にロードするため、メモリは worker 数倍になる。

| コーパス | 1 worker | 2 workers | 4 workers |
|---|---|---|---|
| 1M文 | 3.4 GB | 6.8 GB | 13.6 GB |
| 2M文 | 6.6 GB | 13.2 GB | 26.4 GB |

### 起動コマンド例

```bash
# 開発（1 worker）
uv run python -m web.app

# 本番（4 workers、64GB マシン）
uv run uvicorn web.app:app --host 0.0.0.0 --port 8000 --workers 4
```

## 移行手順（64GB マシンへのスケールアップ）

### 1. 環境セットアップ

```bash
git clone <repo-url>
cd softmatch-db
uv sync
```

### 2. 大規模コーパス取得

```bash
# bench/prepare_corpus.py を修正: max_sentences=2_000_000
uv run python bench/prepare_corpus.py
# → data/wiki_ja.txt (200万行)
```

### 3. FastText 埋め込み取得（語彙数拡大の場合）

```bash
# 現状 50k 語彙で十分だが、100k に拡大する場合:
# bench/prepare_fasttext.py を修正: max_vocab=100_000
uv run python bench/prepare_fasttext.py
```

### 4. インデックス構築（~2.4時間）

```bash
uv run python web/build_index.py
# → web/index.duckdb (~2.4 GB)
```

### 5. 動作確認

```bash
uv run python -m web.app
# http://localhost:8000 で検索テスト
```

## 将来の最適化候補

### ビルド時間短縮
- `SortedIndex.build()` の Python list 化 → NumPy structured array のまま保持
- `np.argsort` → Rust/Cython での並列ソート
- 目標: 2M文のビルドを 2.4hr → 30min 以下

### ランタイムメモリ削減
- SortedIndex を mmap ベースに変更（公式 SoftMatcha2 と同様）
- worker 間でメモリ共有可能に → メモリが worker 数倍にならない
- 目標: 136 bytes/token → ~40 bytes/token

### 検索並列化
- beam search の Cython/Numba 化で GIL 解放
- 1 プロセスで複数リクエストを真に並列処理可能に

## 公式 SoftMatcha2 との比較

| 項目 | SoftMatch-DB (目標) | 公式 SoftMatcha2 |
|---|---|---|
| 言語 | Python + NumPy | Rust |
| トークン規模 | 5000万 | 1.4兆 (28,000倍) |
| インデックス格納 | DuckDB (オンメモリ) | ファイルベース mmap |
| 検索 | 単一プロセス beam search | 分散処理対応 |
| アルゴリズム | 同等 (cand_next + ngram + sorted_index) | 同等 |
| 検索品質 | 公式と同等（MeCab + cand_next で再現） | リファレンス |
