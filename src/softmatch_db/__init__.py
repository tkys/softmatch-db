"""SoftMatch-DB — SoftMatcha2 algorithm as a DuckDB Python extension.

Usage::

    import duckdb
    import softmatch_db

    con = duckdb.connect("index.duckdb")
    softmatch_db.register(con)
    softmatch_db.build(con, "corpus.txt", lang="ja")

    results = softmatch_db.search(con, "金メダリスト")
    print(results)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import duckdb

from softmatch_db.core.searcher import Searcher
from softmatch_db.duckdb_ext import get_searcher, init_searcher, register_udfs
from softmatch_db.index.schema import create_tables

__version__ = "0.1.0"


def register(con: duckdb.DuckDBPyConnection) -> None:
    """Create tables, register UDFs, and optionally initialise the searcher.

    If the vocabulary table already contains data, the in-memory
    ``Searcher`` is initialised immediately.

    Args:
        con: Open DuckDB connection.
    """
    create_tables(con)
    register_udfs(con)

    # Attempt to load searcher if data exists.
    row = con.sql("SELECT COUNT(*) FROM softmatch_vocab").fetchone()
    if row and row[0] > 0:
        init_searcher(con)


def build(
    con: duckdb.DuckDBPyConnection,
    corpus_path: str,
    lang: str = "ja",
    tokenizer: object | None = None,
    embedding: object | None = None,
    pair_cons: int = 2000,
    trio_cons: int = 200,
) -> None:
    """Build the SoftMatch index and re-initialise the searcher.

    Args:
        con: Open DuckDB connection.
        corpus_path: Path to plain-text corpus.
        lang: Language code (``"ja"`` or ``"en"``).
        tokenizer: Tokenizer instance. If ``None``, a default is created
            based on *lang*.
        embedding: Embedding instance. If ``None``, must be provided.
        pair_cons: Pair filter coverage.
        trio_cons: Trio filter coverage.

    Raises:
        ValueError: If no embedding is provided.
    """
    from softmatch_db.index.builder import build_index

    if tokenizer is None:
        if lang == "ja":
            from softmatch_db.tokenizers.mecab_tok import MeCabTokenizer

            tokenizer = MeCabTokenizer.build_from_corpus(corpus_path)
        else:
            from softmatch_db.tokenizers.whitespace_tok import WhitespaceTokenizer

            tokenizer = WhitespaceTokenizer.build_from_corpus(corpus_path)

    if embedding is None:
        raise ValueError("An embedding instance must be provided.")

    build_index(
        con=con,
        corpus_path=corpus_path,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        embedding=embedding,  # type: ignore[arg-type]
        lang=lang,
        pair_cons=pair_cons,
        trio_cons=trio_cons,
    )

    # Re-initialise searcher with new data.
    init_searcher(con)


def search(
    con: duckdb.DuckDBPyConnection,
    query: str,
    top_k: int = 20,
    threshold: float = 0.45,
    max_runtime: float = 10.0,
) -> pd.DataFrame:
    """Run a soft-matching search and return results as a DataFrame.

    The result DataFrame is also registered as ``softmatch_results``
    for SQL access via ``SELECT * FROM softmatch_results``.

    Args:
        con: Open DuckDB connection.
        query: Search query string.
        top_k: Maximum number of results.
        threshold: Minimum similarity threshold.
        max_runtime: Time budget in seconds.

    Returns:
        DataFrame with columns ``rank``, ``tokens``, ``text``,
        ``score``, ``count``.

    Raises:
        RuntimeError: If the searcher is not initialised.
    """
    searcher = get_searcher()
    if searcher is None:
        raise RuntimeError(
            "Searcher not initialised. Call register() or build() first."
        )

    results = searcher.search(
        query=query,
        top_k=top_k,
        min_similarity=threshold,
        max_runtime=max_runtime,
    )

    df = searcher.decode_results(results)

    # Register as a view for SQL access.
    con.register("softmatch_results", df)

    return df
