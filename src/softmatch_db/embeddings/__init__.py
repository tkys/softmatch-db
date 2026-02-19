"""Embedding implementations."""

from softmatch_db.embeddings.base import Embedding
from softmatch_db.embeddings.fasttext_emb import FastTextEmbedding

__all__ = ["Embedding", "FastTextEmbedding"]
