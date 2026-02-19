"""Tests for index/builder.py and index/schema.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pytest

from softmatch_db.embeddings.fasttext_emb import FastTextEmbedding
from softmatch_db.index.builder import build_index
from softmatch_db.index.schema import create_tables
from softmatch_db.tokenizers.whitespace_tok import WhitespaceTokenizer


@pytest.fixture()
def corpus_file(tmp_path: Path) -> str:
    """Create a small English test corpus."""
    text = (
        "the cat sat on the mat\n"
        "a dog ran in the park\n"
        "the sun shines on the hill\n"
        "a bird sang in the tree\n"
        "the wind blows over the sea\n"
    )
    p = tmp_path / "corpus.txt"
    p.write_text(text, encoding="utf-8")
    return str(p)


class TestSchema:
    """Tests for DDL creation."""

    def test_create_tables_idempotent(self) -> None:
        """Creating tables twice should not raise."""
        con = duckdb.connect()
        create_tables(con)
        create_tables(con)
        # Check tables exist.
        tables = [r[0] for r in con.sql("SHOW TABLES").fetchall()]
        assert "softmatch_config" in tables
        assert "softmatch_vocab" in tables
        assert "softmatch_corpus" in tables
        assert "softmatch_ngram_index" in tables
        con.close()


class TestBuildIndex:
    """Tests for the full index build pipeline."""

    def test_build_populates_tables(self, corpus_file: str) -> None:
        """Building an index should populate all tables."""
        con = duckdb.connect()
        tok = WhitespaceTokenizer.build_from_corpus(corpus_file)

        # Create random embeddings matching vocab size.
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((tok.vocab_size, 16)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        emb = FastTextEmbedding.from_array(raw / np.maximum(norms, 1e-12))

        build_index(
            con=con,
            corpus_path=corpus_file,
            tokenizer=tok,
            embedding=emb,
            lang="en",
            pair_cons=50,
            trio_cons=20,
        )

        # Vocab should be populated.
        count = con.sql("SELECT COUNT(*) FROM softmatch_vocab").fetchone()
        assert count is not None and count[0] > 0

        # Corpus should have 5 sentences.
        corpus_count = con.sql("SELECT COUNT(*) FROM softmatch_corpus").fetchone()
        assert corpus_count is not None and corpus_count[0] == 5

        # N-gram index should have pair and trio rows.
        ngram_count = con.sql("SELECT COUNT(*) FROM softmatch_ngram_index").fetchone()
        assert ngram_count is not None and ngram_count[0] == 2

        # Config should have lang.
        lang_row = con.sql(
            "SELECT value FROM softmatch_config WHERE key = 'lang'"
        ).fetchone()
        assert lang_row is not None and lang_row[0] == "en"

        con.close()
