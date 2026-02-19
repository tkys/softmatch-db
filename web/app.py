"""FastAPI web server for SoftMatch-DB search.

Usage::

    uv run python -m web.app          # 0.0.0.0:8000 (LAN accessible)
    uv run uvicorn web.app:app --reload  # development
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import duckdb
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import softmatch_db
from softmatch_db.duckdb_ext import get_searcher

WEB_DIR = Path(__file__).resolve().parent
INDEX_PATH = WEB_DIR / "index.duckdb"
STATIC_DIR = WEB_DIR / "static"

_con: duckdb.DuckDBPyConnection | None = None
_stats: dict[str, int] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open DuckDB connection and initialise the searcher on startup."""
    global _con
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index not found: {INDEX_PATH}. "
            "Run `uv run python web/build_index.py` first."
        )
    _con = duckdb.connect(str(INDEX_PATH))
    softmatch_db.register(_con)

    # Cache dataset stats.
    corpus_row = _con.sql("SELECT COUNT(*) FROM softmatch_corpus").fetchone()
    vocab_row = _con.sql("SELECT COUNT(*) FROM softmatch_vocab").fetchone()
    _stats["corpus_size"] = corpus_row[0] if corpus_row else 0
    _stats["vocab_size"] = vocab_row[0] if vocab_row else 0

    yield
    if _con is not None:
        _con.close()
        _con = None


app = FastAPI(title="SoftMatch-DB Search", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    """Serve the search UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/stats")
async def stats() -> JSONResponse:
    """Return dataset statistics.

    Returns:
        JSON with ``corpus_size`` and ``vocab_size``.
    """
    return JSONResponse(content=_stats)


@app.get("/api/search")
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(20, ge=1, le=100, description="Max results"),
    threshold: float = Query(0.45, ge=0.0, le=1.0, description="Min similarity"),
) -> JSONResponse:
    """Run a soft-matching search and return JSON results.

    Args:
        q: Search query string.
        top_k: Maximum number of results to return.
        threshold: Minimum similarity score threshold.

    Returns:
        JSON with ``query``, ``results``, ``count``, and ``elapsed_ms``.
    """
    if _con is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Index not loaded"},
        )

    t0 = time.perf_counter()
    df = softmatch_db.search(_con, q, top_k=top_k, threshold=threshold)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    results = []
    for _, row in df.iterrows():
        results.append({
            "rank": int(row["rank"]),
            "text": str(row["text"]),
            "score": round(float(row["score"]), 4),
            "count": int(row["count"]),
            "token_ids": [int(t) for t in row["tokens"]],
        })

    return JSONResponse(content={
        "query": q,
        "results": results,
        "count": len(results),
        "elapsed_ms": round(elapsed_ms, 1),
    })


@app.get("/api/kwic")
async def kwic(
    tokens: str = Query(..., description="Comma-separated token IDs"),
    max: int = Query(5, ge=1, le=20, description="Max examples"),
) -> JSONResponse:
    """Find corpus sentences containing the given token subsequence (KWIC).

    Args:
        tokens: Comma-separated token IDs (e.g. ``"694,210"``).
        max: Maximum number of example sentences.

    Returns:
        JSON with ``examples`` list and ``count``.
    """
    if _con is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Index not loaded"},
        )

    searcher = get_searcher()
    if searcher is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Searcher not initialised"},
        )

    try:
        token_ids = [int(t.strip()) for t in tokens.split(",") if t.strip()]
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid token IDs"},
        )

    if not token_ids:
        return JSONResponse(content={"examples": [], "count": 0})

    examples = searcher.find_examples(_con, token_ids, max_examples=max)

    return JSONResponse(content={
        "examples": examples,
        "count": len(examples),
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "web.app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
