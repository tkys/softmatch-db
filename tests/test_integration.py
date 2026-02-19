"""End-to-end integration tests."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pytest

import softmatch_db
from softmatch_db.embeddings.fasttext_emb import FastTextEmbedding
from softmatch_db.tokenizers.whitespace_tok import WhitespaceTokenizer


@pytest.fixture()
def corpus_file(tmp_path: Path) -> str:
    """Create a small controlled English corpus for integration testing."""
    lines = [
        "gold medalist won the competition",
        "silver medalist also performed well",
        "the champion earned a gold medal",
        "olympic athlete achieved gold medal",
        "gold medal winner became a hero",
        "silver medal winner was also praised",
        "the gold medalist set a new record",
        "world championship gold medal victory",
        "tokyo olympic gold medalist retired",
        "gold medal dreams of young athletes",
        "bronze medalist also fought well",
        "olympic athlete efforts are immeasurable",
        "gold medalist interview went viral",
        "silver medalist aims for gold next time",
        "japanese athlete set world record",
    ]
    p = tmp_path / "corpus_en.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


@pytest.fixture()
def db_with_index(corpus_file: str) -> duckdb.DuckDBPyConnection:
    """Build an in-memory DuckDB with index."""
    con = duckdb.connect()
    tok = WhitespaceTokenizer.build_from_corpus(corpus_file)

    rng = np.random.default_rng(42)
    raw = rng.standard_normal((tok.vocab_size, 16)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    emb = FastTextEmbedding.from_array(raw / np.maximum(norms, 1e-12))

    softmatch_db.register(con)
    softmatch_db.build(
        con=con,
        corpus_path=corpus_file,
        lang="en",
        tokenizer=tok,
        embedding=emb,
        pair_cons=50,
        trio_cons=20,
    )
    return con


class TestIntegration:
    """End-to-end integration tests."""

    def test_search_returns_dataframe(
        self, db_with_index: duckdb.DuckDBPyConnection
    ) -> None:
        """search() should return a non-empty DataFrame."""
        df = softmatch_db.search(
            db_with_index,
            "gold medalist",
            top_k=5,
            threshold=0.2,
            max_runtime=5.0,
        )
        assert not df.empty
        assert "score" in df.columns
        assert "text" in df.columns

    def test_search_results_accessible_via_sql(
        self, db_with_index: duckdb.DuckDBPyConnection
    ) -> None:
        """After search(), results should be queryable via SQL."""
        softmatch_db.search(
            db_with_index, "gold medalist", top_k=5, threshold=0.2
        )
        result = db_with_index.sql(
            "SELECT * FROM softmatch_results"
        ).fetchdf()
        assert not result.empty

    def test_sql_tables_exist(
        self, db_with_index: duckdb.DuckDBPyConnection
    ) -> None:
        """All expected tables should exist after build."""
        tables = [
            r[0]
            for r in db_with_index.sql("SHOW TABLES").fetchall()
        ]
        for name in [
            "softmatch_config",
            "softmatch_vocab",
            "softmatch_corpus",
            "softmatch_ngram_index",
        ]:
            assert name in tables
