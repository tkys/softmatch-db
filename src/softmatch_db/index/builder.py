"""Index builder — tokenise corpus and populate DuckDB tables."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from softmatch_db.core.ngram_filter import NgramBitFilter
from softmatch_db.core.zipfian import compute_zipfian_norms
from softmatch_db.index.schema import create_tables

if TYPE_CHECKING:
    import duckdb

logger = logging.getLogger(__name__)

NDArrayF32 = NDArray[np.float32]


@runtime_checkable
class TokenizerLike(Protocol):
    """Minimal tokenizer interface needed by the builder."""

    def tokenize(self, line: str) -> list[str]: ...
    def encode(self, tokens: list[str]) -> list[int]: ...

    @property
    def vocab_size(self) -> int: ...


@runtime_checkable
class EmbeddingLike(Protocol):
    """Minimal embedding interface needed by the builder."""

    @property
    def embeddings(self) -> NDArrayF32: ...


def build_index(
    con: duckdb.DuckDBPyConnection,
    corpus_path: str,
    tokenizer: TokenizerLike,
    embedding: EmbeddingLike,
    lang: str = "ja",
    pair_cons: int = 2000,
    trio_cons: int = 200,
) -> None:
    """Build the SoftMatch-DB index from a text corpus.

    Args:
        con: Open DuckDB connection (can be in-memory or persistent).
        corpus_path: Path to plain-text corpus (one sentence per line).
        tokenizer: Tokenizer instance satisfying ``TokenizerLike``.
        embedding: Embedding instance satisfying ``EmbeddingLike``.
        lang: Language code stored in config.
        pair_cons: Number of top-frequency tokens tracked in pair filter.
        trio_cons: Number of top-frequency tokens tracked in trio filter.
    """
    logger.info("Creating tables...")
    create_tables(con)

    # ------------------------------------------------------------------
    # 1. Read and tokenise corpus
    # ------------------------------------------------------------------
    logger.info("Tokenising corpus: %s", corpus_path)
    all_lines: list[str] = []
    all_token_strs: list[list[str]] = []
    all_token_ids: list[list[int]] = []

    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            toks = tokenizer.tokenize(line)
            ids = tokenizer.encode(toks)
            all_lines.append(line)
            all_token_strs.append(toks)
            all_token_ids.append(ids)

    # ------------------------------------------------------------------
    # 2. Frequency counts (vectorized)
    # ------------------------------------------------------------------
    flat_tokens = np.array(
        [t for ids in all_token_ids for t in ids], dtype=np.uint32
    )
    freq = np.bincount(
        flat_tokens[flat_tokens < tokenizer.vocab_size],
        minlength=tokenizer.vocab_size,
    ).astype(np.int64)

    # ------------------------------------------------------------------
    # 3. Populate softmatch_vocab (bulk DataFrame insert)
    # ------------------------------------------------------------------
    logger.info("Populating vocabulary table (%d tokens)...", tokenizer.vocab_size)
    emb_matrix = embedding.embeddings
    v_count = min(tokenizer.vocab_size, len(emb_matrix))

    con.execute("DELETE FROM softmatch_vocab")
    _insert_vocab_bulk(con, tokenizer, emb_matrix, freq, v_count)

    # ------------------------------------------------------------------
    # 4. Compute Zipfian norms and batch-update
    # ------------------------------------------------------------------
    logger.info("Computing Zipfian norms...")
    norm_sq = compute_zipfian_norms(
        emb_matrix[:v_count], freq[:v_count]
    )
    norm_df = pd.DataFrame({
        "token_id": np.arange(v_count, dtype=np.int32),
        "norm_sq": norm_sq[:v_count].astype(float),
    })
    con.register("_norm_tmp", norm_df)
    con.execute("""
        UPDATE softmatch_vocab
        SET norm_sq = _norm_tmp.norm_sq
        FROM _norm_tmp
        WHERE softmatch_vocab.token_id = _norm_tmp.token_id
    """)
    con.unregister("_norm_tmp")

    # ------------------------------------------------------------------
    # 5. Build N-gram filter
    # ------------------------------------------------------------------
    logger.info("Building N-gram filter (pair_cons=%d, trio_cons=%d)...",
                pair_cons, trio_cons)
    ngf = NgramBitFilter.build_from_corpus(flat_tokens, pair_cons, trio_cons)
    pair_blob, trio_blob = ngf.to_bytes()

    con.execute("DELETE FROM softmatch_ngram_index")
    con.execute(
        "INSERT INTO softmatch_ngram_index VALUES (?, ?, ?)",
        ["pair", pair_cons, pair_blob],
    )
    con.execute(
        "INSERT INTO softmatch_ngram_index VALUES (?, ?, ?)",
        ["trio", trio_cons, trio_blob],
    )

    # ------------------------------------------------------------------
    # 6. Populate corpus table (bulk DataFrame insert)
    # ------------------------------------------------------------------
    logger.info("Populating corpus table (%d sentences)...", len(all_lines))
    con.execute("DELETE FROM softmatch_corpus")
    _insert_corpus_bulk(con, all_lines, all_token_ids)

    # ------------------------------------------------------------------
    # 7. Config
    # ------------------------------------------------------------------
    con.execute("DELETE FROM softmatch_config")
    con.execute(
        "INSERT INTO softmatch_config VALUES ('lang', ?)", [lang]
    )
    con.execute(
        "INSERT INTO softmatch_config VALUES ('embedding_dim', ?)",
        [str(emb_matrix.shape[1])],
    )

    logger.info("Index build complete.")


def _insert_vocab_bulk(
    con: duckdb.DuckDBPyConnection,
    tokenizer: TokenizerLike,
    emb_matrix: NDArrayF32,
    freq: NDArray[np.int64],
    v_count: int,
) -> None:
    """Insert vocabulary rows via bulk DataFrame registration.

    Args:
        con: DuckDB connection.
        tokenizer: Tokenizer instance.
        emb_matrix: Embedding matrix.
        freq: Frequency array.
        v_count: Number of vocab entries to insert.
    """
    # Build id→token mapping.
    id_to_token: dict[int, str] | None = None
    if hasattr(tokenizer, "_id_to_token"):
        id_to_token = tokenizer._id_to_token
    elif hasattr(tokenizer, "decode"):
        id_to_token = {}
        for tid in range(v_count):
            decoded = tokenizer.decode([tid])  # type: ignore[attr-defined]
            id_to_token[tid] = decoded[0] if decoded else f"<{tid}>"

    if id_to_token is None:
        id_to_token = {i: f"<{i}>" for i in range(v_count)}

    tokens_list = [id_to_token.get(i, f"<{i}>") for i in range(v_count)]
    emb_list = [emb_matrix[i].tolist() for i in range(v_count)]

    vocab_df = pd.DataFrame({
        "token_id": np.arange(v_count, dtype=np.int32),
        "token": tokens_list,
        "freq": freq[:v_count].astype(np.int64),
        "norm_sq": np.full(v_count, 1e10, dtype=np.float64),
        "embedding": emb_list,
    })
    con.register("_vocab_tmp", vocab_df)
    con.execute(
        "INSERT INTO softmatch_vocab "
        "SELECT token_id, token, freq, norm_sq, embedding FROM _vocab_tmp"
    )
    con.unregister("_vocab_tmp")


def _insert_corpus_bulk(
    con: duckdb.DuckDBPyConnection,
    all_lines: list[str],
    all_token_ids: list[list[int]],
) -> None:
    """Insert corpus rows via bulk DataFrame registration.

    Args:
        con: DuckDB connection.
        all_lines: List of sentence texts.
        all_token_ids: List of token id sequences.
    """
    n = len(all_lines)
    byte_lengths = np.array(
        [len(line.encode("utf-8")) + 1 for line in all_lines],
        dtype=np.int64,
    )
    byte_starts = np.zeros(n, dtype=np.int64)
    if n > 1:
        np.cumsum(byte_lengths[:-1], out=byte_starts[1:])

    corpus_df = pd.DataFrame({
        "doc_id": np.zeros(n, dtype=np.int32),
        "sent_id": np.arange(n, dtype=np.int32),
        "text": all_lines,
        "byte_start": byte_starts,
        "tokens": all_token_ids,
    })
    con.register("_corpus_tmp", corpus_df)
    con.execute(
        "INSERT INTO softmatch_corpus "
        "SELECT doc_id, sent_id, text, byte_start, tokens FROM _corpus_tmp"
    )
    con.unregister("_corpus_tmp")
