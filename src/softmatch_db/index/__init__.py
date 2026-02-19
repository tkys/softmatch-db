"""Index building and schema management."""

from softmatch_db.index.builder import build_index
from softmatch_db.index.schema import create_tables

__all__ = ["build_index", "create_tables"]
