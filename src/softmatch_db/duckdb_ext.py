"""DuckDB extension registration — UDF and helper functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb

from softmatch_db.core.searcher import Searcher

# Module-level searcher singleton (lazily initialised).
_searcher: Searcher | None = None


def _get_searcher() -> Searcher:
    """Return the module-level Searcher, raising if uninitialised."""
    if _searcher is None:
        raise RuntimeError(
            "Searcher not initialised. Call softmatch_db.register(con) first."
        )
    return _searcher


def register_udfs(con: duckdb.DuckDBPyConnection) -> None:
    """Register scalar UDFs with the DuckDB connection.

    Args:
        con: DuckDB connection.
    """
    import duckdb as _duckdb

    def softmatch_score(query: str, tokens: list[int]) -> float:
        """Scalar UDF: compute softmin score between query and tokens."""
        searcher = _get_searcher()
        return searcher.score(query, tokens)

    con.create_function(
        "softmatch_score",
        softmatch_score,
        ["VARCHAR", "INTEGER[]"],
        "FLOAT",
    )


def init_searcher(con: duckdb.DuckDBPyConnection) -> Searcher:
    """Initialise the module-level Searcher from DuckDB data.

    Args:
        con: DuckDB connection with populated softmatch tables.

    Returns:
        The initialised ``Searcher``.
    """
    global _searcher
    _searcher = Searcher(con)
    return _searcher


def get_searcher() -> Searcher | None:
    """Return current searcher instance (may be None)."""
    return _searcher
