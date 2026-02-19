# SoftMatch-DB PROJECT_LOG

---

## 2026-02-19

### 達成したこと

#### 実装完了（全13ステップ）
- `core/softmin.py` — SoftMin 関数（スカラー・ベクトル化版）
- `core/zipfian.py` — Zipfian Whitening（頻度加重 PCA 白色化）
- `core/ngram_filter.py` — N-gram ビットフィルタ（pair/trio の O(1) bitset ルックアップ）
- `core/sorted_index.py` — 簡略サフィックス配列（4×u64 圧縮ハッシュ + bisect 二分探索）
- `core/beam_search.py` — ビームサーチ本体（match/delete/insert, Pareto 枝刈り）
- `core/searcher.py` — 統合 Searcher クラス
- `tokenizers/` — WhitespaceTokenizer, SudachiTokenizer (SplitMode.C)
- `embeddings/` — FastTextEmbedding (.npy 読み込み)
- `index/schema.py` + `index/builder.py` — DuckDB テーブル定義 + インデックス構築
- `duckdb_ext.py` + `__init__.py` — DuckDB スカラー UDF + Python API
- テスト 61 本全通過（unit + integration + perf）

#### バグ修正
- DuckDB `typing.VARCHAR` → 文字列型名 `"VARCHAR"` に変更（DuckDB 1.4.4 非互換）
- `Searcher.search()` の日本語トークナイズ不具合修正：`_build_tokenize_fn()` で SudachiPy を使うよう修正
- `builder.py` の語彙・コーパス INSERT を 1 行ずつ → `executemany` バッチ化

#### ベンチマーク基盤整備
- `bench/prepare_corpus.py` — HuggingFace `wikimedia/wikipedia` から Wikipedia 日本語を 1 文 1 行で取得
- `bench/prepare_fasttext.py` — FastText cc.ja.300 を `.vec.gz` 形式で取得・`.npy` 変換（50k語彙, 60MB）
- `bench/run_bench.py` — E2E ベンチ：フェーズ別計測（sim_matrix / beam / decode）+ JSON 出力
- `tests/test_perf.py` — pytest-benchmark 16 本（合成高スコア score_matrix でビームサーチ負荷を再現）

### 決定事項
- FastText は `.bin`（4.2GB）より軽量な `.vec.gz`（1.4GB）形式で取得
- Wikipedia コーパスは `wikimedia/wikipedia 20231101.ja`（streaming）から 50k 文を抽出
- DuckDB 型は文字列で指定（`"VARCHAR"`, `"INTEGER[]"`, `"FLOAT"`）
- バッチ INSERT サイズ：語彙 500 件 / コーパス 500 件

### ベンチマーク結果（FastText 300d × Wikipedia 日本語）

#### 構築性能
| 規模 | 語彙 | トークン | 構築時間 | ピークメモリ |
|------|------|---------|---------|------------|
| 5k行 | 10k | 147k | 33.1s | 163.8 MB |
| 10k行 | 20k | 285k | 65.2s | 323.6 MB |
| 20k行 | 30k | 529k | 108.6s | 495.4 MB |
| 50k行 | 50k | 1.2M | 217.2s | 861.1 MB |

#### 検索性能（best ms / 3 run）
| クエリ | 5k行 | 10k行 | 20k行 | 50k行 |
|--------|------|-------|-------|-------|
| 金メダリスト | 19.4 | 35.0 | 52.7 | 97.8 |
| オリンピック選手 | 17.8 | 35.9 | 53.5 | 86.4 |
| 東京大学 | 13.3 | 26.0 | 39.2 | 65.0 |
| 人工知能 | 12.6 | 25.0 | 37.6 | 63.6 |
| 新幹線 | 13.9 | 25.6 | 38.6 | 65.2 |

#### 50k行内訳（平均）
- similarity matrix: ~28ms（語彙50k×300次元行列積）
- beam search: ~54ms（支配的ボトルネック）
- decode: ~0.7ms

#### pytest-benchmark 代表値（unit レベル）
| テスト | 値 |
|--------|-----|
| has_pair lookup | ~315 ns |
| softmin scalar | ~320 ns |
| exists_cached | ~755 ns |
| exists_no_cache | ~4 µs |
| softmin_vec 10k | ~119 µs |
| beam_search 3-tok (合成) | ~2.3 ms |
| SortedIndex build 50k | ~208 ms |
| NgramFilter build 100k | ~74 ms |

### パフォーマンス最適化（5段階）

#### Step 1: DataFrame bulk INSERT（builder.py）
- `executemany` バッチ → `pd.DataFrame` + `con.register()` + `INSERT INTO ... SELECT FROM`
- 頻度カウント: 二重 Python ループ → `np.bincount`

#### Step 2: SortedIndex.build() NumPy ベクトル化
- `as_strided()` で 12-token sliding window のゼロコピービュー生成
- ハッシュ計算を全ポジション同時に NumPy ビット演算
- `np.argsort` + structured array で lexicographic sort

#### Step 3: NgramBitFilter.build_from_corpus() NumPy ベクトル化
- 2-gram/3-gram のハッシュ計算を全コーパス位置で一括実行
- `np.unique` で deduplicate → ビット設定ループは unique ハッシュのみ

#### Step 4: DuckDB list_cosine_similarity 評価 → 却下
- DuckDB の `list_dot_product` は NumPy BLAS matmul の **357 倍遅い**（1287ms vs 3.6ms）
- NumPy アプローチを維持

#### Step 5: beam_search 前処理ベクトル化 + 正規化キャッシュ
- `match_sim` 構築: P×V Python ループ → `np.nonzero` + `np.argsort`（位置ごと）
- `inst_mult` 構築: ソート・フィルタを NumPy mask + argsort
- `_compute_similarity()`: 語彙正規化を Searcher 初期化時に 1 回実行、キャッシュ再利用

### 最適化結果（50k コーパス）

| バージョン | ビルド | 検索avg(best) | sim_matrix | beam | 主な変更 |
|---|---|---|---|---|---|
| v1 初期 | 217.2s | 75.6ms | ~27ms | ~50ms | ベースライン |
| v2 Step1-3 | 36.9s | 75.6ms | ~27ms | ~50ms | bulk INSERT + NumPy vectorize |
| v3 Step5前半 | 36.7s | 30.2ms | ~27ms | 5.6ms | beam前処理vectorize |
| **v4 Step5完了** | **36.8s** | **8.2ms** | **2.7ms** | **5.2ms** | **正規化キャッシュ** |

**改善率**: ビルド **5.9x**、検索 **9.2x** 高速化

### 次の課題 (Next Steps)
- **ビルド高速化**: SortedIndex.build() の tolist() + Python list 化（~20s）がまだ支配的 → NumPy array のまま bisect 代替を検討
- **語彙カバレッジ向上**: FastText と SudachiPy の語彙ミスマッチ（現在 46%）→ 表層形正規化で改善
- **大規模コーパス対応**: 50k 行超でのメモリ管理（現在 866MB peak）
- **DuckDB 永続化テスト**: インメモリ以外でのロード/クエリの動作確認
- **_enumerate 内部ループ最適化**: beam_search のコアループ（softmin + ngram_filter + sorted_index.exists）は依然 Python — Cython/Numba 化の余地

---

## 2026-02-19 (2) — Web 検索アプリ構築 + 公式差異分析

### 達成したこと

#### Web 検索 UI 実装
- `web/app.py` — FastAPI サーバー（lifespan で DuckDB + Searcher 初期化、`0.0.0.0:8000` バインド）
- `web/build_index.py` — `data/wiki_ja.txt` + `data/fasttext_ja/embeddings.npy` → `web/index.duckdb` 構築
- `web/static/index.html` — vanilla HTML/JS/CSS 単一ファイル検索 UI（レスポンシブ、日本語フォント対応）
- `/api/search?q=...&top_k=20&threshold=0.55` → JSON レスポンス（query, results, count, elapsed_ms）
- `/api/stats` → データセット統計（corpus_size: 50,000 / vocab_size: 50,000）
- 依存追加: `fastapi`, `uvicorn[standard]`

#### 動作確認結果
- インデックス構築: 50,000語彙 / 50,000文（正常完了）
- 検索テスト: `金メダリスト` → 3件ヒット、52.9ms
- LAN アクセス対応（`host=0.0.0.0`）

### 問題発見: 公式 SoftMatcha2 との検索品質乖離

ブラウザで「京都」を検索すると、完全一致の「京都」(score=1.0, count=78) に加えて「努力」(0.52)、「興業」(0.51)、「宇部」(0.49) など**意味的に無関係なトークン**がヒットした。公式 SoftMatcha2 ではこのようなノイズは出ない。

### 原因分析: 公式 Rust ソースとの比較

公式リポジトリ `softmatcha/softmatcha2` の Rust ソース (`rust/src/search/`) を精読し、現実装との差異を特定した。

#### 公式の3層フィルタリング構造

| 層 | メカニズム | コード箇所 | 現実装 |
|---|---|---|---|
| **1. `cand_next` 先読み** | ソート済みインデックスファイルから、現在のprefixの**次に実際にコーパスで出現するトークン**だけを候補に絞る | `z_enumerate.rs:107-129` | **欠落**（beam_search.py のコメントで明示的に省略） |
| **2. `check_valid` n-gram** | 2-gram/3-gram ビットセットでコーパス存在チェック | `z_check.rs:19-34` | あり（同等の実装） |
| **3. `get_match_exists_main`** | サフィックス配列バイナリサーチで候補シーケンスの完全一致確認 | `z_enumerate.rs:215-220` | `sorted_index.exists()` で同等 |

#### `cand_next` の仕組み（核心部分）

```rust
// z_enumerate.rs:109-128
// 現在のprefix(current_seq)のソート済みインデックス位置(start_pos)から
// 最大50エントリを読み取り、次に来るトークンIDを抽出
let target_pos: usize = idx_length.min(*start_pos + 50);
let buf = read_from_file(&file, stt_bytes, end_bytes);
for i in *start_pos..target_pos {
    let cnd = retrieve_value(&tgt, current_seq.len()) as u32;
    cand_next.push((cnd, (i - start_pos) as i32));
}
```

matching / insertion の両方で:
```rust
// z_enumerate.rs:137-138, 170
let v = bsearch(&cand_next, *k as u32);
if cand_next.len() == 0 || v != -1 {
    // cand_next が空なら check_valid フォールバック
    // cand_next が非空なら、その中に存在する候補のみ許可
```

#### なぜ1トークンクエリで特に問題が顕著か

1. **`check_valid`**: `seq.len() >= 2` でのみ発動 → 1トークンでは**無効**
2. **`sorted_index.exists([v])`**: コーパスに存在する全トークンで `True` → **意味なし**
3. **`cand_next`**: 省略されているため**不在**
4. → **コサイン類似度閾値のみ**が残るため、FastText空間でたまたま近い無関係トークンが通過

#### 公式で「京都」→「努力」が出ない理由

公式では空prefixから `[京都]` を生成した時点で `get_match_exists_main` がサフィックス配列の**位置 (posi)** を返す。次のステップで `cand_next` はその `posi` 付近のインデックスエントリから**実際にコーパスで後続するトークン**だけを読み取る。「京都」の後に「努力」が来る文はコーパスに存在しないため、`cand_next` に入らず候補にならない。

また1トークン結果だけを見ても、`sim * mult` スコアで「京都」(1.0) が圧倒的上位となり、0.52程度のノイズは `top_k` カットでほぼ除外される（公式はこの段階でもっと多くの質の高い多トークン候補がある）。

### 現実装の `beam_search.py` で省略された理由（推測）

beam_search.py の冒頭コメント:
> "The file-based candidate lookahead optimisation from z_enumerate.rs:108-129 is omitted"

公式はソート済みインデックスを**ファイルベースの mmap** で管理し、任意位置を O(1) で読み取れる。現実装は `SortedIndex` を Python list (`self.hashes`, `self.positions`) として全メモリに保持しているため、同じ「位置ベースの先読み」を実装するにはデータ構造の設計変更が必要。

### 決定事項
- デフォルト threshold を `0.45` → `0.55` に引き上げ（暫定対処）
- UI にデータセット統計を表示（corpus_size / vocab_size）

### 次の課題 (Next Steps)

#### 検索品質改善（優先度高）
- **`cand_next` 先読みの実装**: `SortedIndex` に「指定位置から N エントリ先読みして次トークンを返す」メソッドを追加 → `beam_search.py` の `_enumerate` ループに組み込む
  - `SortedIndex.positions` リストがあるのでインデックス位置から逆引きは可能
  - `retrieve_value` 相当の処理（圧縮ハッシュからトークンID抽出）の Python 実装が必要
- **`start_pos` の持ち回し**: 公式は `(sim, mult, seq, start_pos)` の4タプルで候補を管理。現実装は `(sim, mult, seq)` の3タプルなので拡張が必要

#### 表示品質改善
- **KWIC（Keyword-In-Context）表示**: マッチしたパターンの出現位置からコーパスの元文を引いて前後文脈を表示
  - `SortedIndex.positions` → コーパス内バイト位置 → `softmatch_corpus` テーブルの `byte_start` で逆引き
  - 公式の `softmatcha-exact` コマンド相当の機能
- **2段階ワークフロー**: search（バリアント一覧）→ exact（個別出現例）の UI 導線

#### 既存課題（前回から継続）
- ビルド高速化: SortedIndex.build() の tolist() 改善
- 語彙カバレッジ: FastText/SudachiPy ミスマッチ（46%）
- _enumerate 内部ループの Cython/Numba 化

---

## 2026-02-19 (3) — cand_next 実装 + MeCab トークナイザ移行

### 達成したこと

#### cand_next サフィックス配列先読みの実装
- `sorted_index.py` に4メソッド追加:
  - `retrieve_token_at()` — 圧縮ハッシュから指定位置のトークンID抽出（`helper.rs:75-115` 移植）
  - `get_start_pos()` — シーケンスのソート済みインデックス下限位置を返却
  - `get_next_tokens()` — start_pos から window 件走査し次トークン候補を抽出（`z_enumerate.rs:108-129` 移植）
  - `_bsearch_token()` — cand_next 内でトークンIDを二分探索（`helper.rs:276-294` 移植）
- `beam_search.py` を全面改修:
  - 候補タプル: `(sim, mult, seq)` → `(sim, mult, seq, start_pos)` に拡張
  - Match/Insert ロジックに cand_next 分岐を追加（先読みあり → bsearch、なし → check_valid フォールバック）
  - `pareto_prune` も4タプル対応
- テスト19本追加（`tests/test_cand_next.py`）、全通過

#### MeCab トークナイザへの移行（根本原因の解消）
- **根本原因の特定**: cand_next 実装後も1トークンクエリのノイズが消えなかった原因は **SudachiPy SplitMode.C のトークン粒度**
  - SplitMode.C は複合語を保持: `東京大学` → `["東京大学"]` (1トークン)
  - 1トークンクエリでは cand_next/check_valid/exists すべて無効化
- **公式トークナイザの特定**: 公式 SoftMatcha2 は **MeCab (ipadic, `-Owakati`)** を使用
  - MeCab は複合語を分割: `人工知能` → `["人工", "知能"]`
  - FastText cc.ja.300 は MeCab で学習されているため語彙カバレッジが高い
- **実装**:
  - `tokenizers/mecab_tok.py` — MeCab/ipadic トークナイザ（`from_fasttext_vocab` + `build_from_corpus`）
  - `core/searcher.py` — クエリ時トークナイザを MeCab 優先に変更（SudachiPy フォールバック）
  - `__init__.py` — デフォルトビルドトークナイザを MeCab に変更
  - `web/build_index.py` — FastText vocab.json から直接語彙構築
  - 依存追加: `mecab-python3`, `ipadic`

### 検索品質改善結果

| クエリ | Before (SudachiPy) | After (MeCab) |
|---|---|---|
| 京都 | 京都(1.0), **努力**(0.52), **興業**(0.51) | 京都(1.0), **大阪**(0.70), **奈良**(0.66), **府**(0.65) |
| 東京大学 | 1トークンで結果薄い | **早稲田大学**(0.72), **九州大学**(0.70), **東北大学**(0.69) |
| 新幹線 | ノイズ混在 | **東海道新幹線**(0.76), **東北新幹線**(0.68), **電車**(0.57) |
| 人工知能 | vocab外(UNK) → 結果なし | **人工 知能**(1.00) |

**ノイズが完全に消え、意味的に関連するパターンのみ返却。**

### 技術的知見

1. **トークナイザと埋め込みの語彙一致が最重要**: FastText cc.ja.300 は MeCab でトークン化されたテキストで学習。SudachiPy の語彙とは46%がミスマッチだった。MeCab 使用で語彙カバレッジが大幅改善。
2. **cand_next は多トークンシーケンスで威力を発揮**: 空 prefix → 1トークン目は window 超過で先読み無効だが、2トークン目以降はコーパスに実在する後続トークンのみに候補を絞る。
3. **3層フィルタリングの相乗効果**: cand_next + check_valid + exists が全て正しく機能するには、トークナイザがマルチトークンシーケンスを生成する必要がある。

### テスト状況
- 全80テスト通過（unit 64 + benchmark 16）
- cand_next テスト19本含む

### 次の課題 (Next Steps)
- ~~**KWIC（Keyword-In-Context）表示**: マッチパターンの出現コンテキスト表示~~ → 完了 (4)
- **ビルド高速化**: SortedIndex.build() の tolist() 改善
- **_enumerate 内部ループ最適化**: Cython/Numba 化の余地
- **大規模コーパス対応**: 50k行超でのメモリ管理

---

## 2026-02-19 (4) — Web UI 改善 + KWIC 出現例表示

### 達成したこと

#### Searcher に KWIC 検索メソッド追加 (`core/searcher.py`)
- `find_examples(con, token_ids, max_examples=5)` — 指定トークン列を含むコーパス文を検索
  - DuckDB `list_contains` でプリフィルタ → Python で連続部分列マッチを検証
  - マッチトークンの表層を連結し `str.find()` で原文中の正確な文字オフセットを算出
  - 返却: `[{text, sent_id, tokens, match_start, match_end}]`
- `_find_subsequence(haystack, needle)` — リスト中の連続部分列検索（静的メソッド）

#### API 改善 (`web/app.py`)
- `/api/search` レスポンスに `token_ids` フィールド追加（KWIC 呼び出し用）
- `/api/kwic?tokens=694,210&max=5` エンドポイント新設
  - カンマ区切りトークンID → `Searcher.find_examples()` に委譲
  - バリデーション付き（不正 ID → 400 エラー）

#### フロントエンド大幅改善 (`web/static/index.html`)
- **KWIC 展開表示**: 結果行クリック → `/api/kwic` を遅延フェッチ → 出現例を `<mark>` ハイライト付きで展開
  - クライアント側キャッシュ（同一トークン列の再フェッチ回避）
  - 展開アイコン ▶ が回転して開閉状態を表示
- **サジェストチップ**: 「京都」「新幹線」「人工知能」「オリンピック」をクリックで即検索
- **threshold デフォルト統一**: HTML 側 0.55 → **0.45**（API 側と一致）
- **スコアバー色段階**: 0.8+ 青、0.6-0.8 緑、0.45-0.6 黄
- **トークン区切り表示**: `人工|知能` のようにパイプ区切りで視覚化
- **空状態改善**: 初回表示時にサンプルクエリの案内を表示

#### バグ修正: KWIC ハイライト位置ずれ
- **原因**: 日本語テキスト（スペースなし）に対してスペース結合前提のオフセット計算をしていた
  - 旧: `char_start = sum(len(s) for surfaces[:match_pos]) + match_pos`（スペース分加算）
- **修正**: マッチトークン表層を連結し `str.find()` で原文中の位置を直接検索
  - 日本語（スペースなし）→ `"".join(surfaces)` で検索
  - 英語フォールバック → `" ".join(surfaces)` で検索

### 決定事項
- KWIC のハイライト位置計算は原文テキスト中の `str.find()` に基づく（トークン位置の積算は不正確）
- KWIC 最大表示件数は 5 件（API の `max` パラメータで変更可）
- フロントエンドの KWIC キャッシュはセッション内メモリ（ページリロードでクリア）

### テスト状況
- 全80テスト通過（unit 64 + benchmark 16）
- 既存テストへの影響なし

### 次の課題 (Next Steps)
- **スケールアップ**: 64GB マシンへ移行し 200万文 / 5000万トークンで再構築（詳細は `docs/SCALING.md`）
- **ビルド高速化**: SortedIndex.build() の tolist() 改善
- **_enumerate 内部ループ最適化**: Cython/Numba 化の余地
- **KWIC の改善案**: 出現件数の全数カウント表示、ページング対応

---

## 2026-02-19 (5) — スケーリング分析 + GitHub 公開

### スケーリング分析

公式 SoftMatcha2（1.4T トークン）に対し、単一マシン Pure Python でどこまでスケールできるか分析。
詳細は `docs/SCALING.md` に記載。

#### ボトルネック整理

| 観点 | ボトルネック | 根拠 |
|---|---|---|
| ビルド | **時間**（Pure Python の線形スケール） | SortedIndex.build() の argsort + Python list 化 |
| ランタイム RAM | **SortedIndex: 136 bytes/token** | 4×u64 ハッシュ + positions リスト |
| 検索レイテンシ | 問題なし | bisect O(log N)、50k→1M で 8ms→15ms 程度 |
| 同時リクエスト | **メモリ × worker 数** | GIL により worker プロセス増が必要 |

#### 決定事項
- **目標規模**: 200万文 / 5000万トークン（64GB マシン前提）
- 「Wikipedia 日本語 200万文」はプロジェクトとしての見栄えが良い
- ビルド ~2.4 時間（1回きり）、ランタイム RAM ~6.6GB、4 workers でも 26.4GB で 64GB に収まる
