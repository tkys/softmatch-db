"""DuckDB schema definitions for SoftMatch-DB."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb


def create_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create all required tables if they don't already exist.

    Args:
        con: Open DuckDB connection.
    """
    con.execute("""
        CREATE TABLE IF NOT EXISTS softmatch_config (
            key    VARCHAR PRIMARY KEY,
            value  VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS softmatch_vocab (
            token_id   INTEGER PRIMARY KEY,
            token      VARCHAR NOT NULL,
            freq       BIGINT  NOT NULL DEFAULT 0,
            norm_sq    FLOAT   NOT NULL DEFAULT 1e10,
            embedding  FLOAT[] NOT NULL
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS softmatch_corpus (
            doc_id     INTEGER NOT NULL,
            sent_id    INTEGER NOT NULL,
            text       VARCHAR NOT NULL,
            byte_start BIGINT,
            tokens     INTEGER[] NOT NULL,
            PRIMARY KEY (doc_id, sent_id)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS softmatch_ngram_index (
            ngram_type VARCHAR PRIMARY KEY,
            cons       INTEGER NOT NULL,
            bits       BLOB NOT NULL
        )
    """)
