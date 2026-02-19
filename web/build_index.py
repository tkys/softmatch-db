"""Build a persistent DuckDB index for the web search app.

Usage::

    uv run python web/build_index.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

import softmatch_db
from softmatch_db.embeddings.fasttext_emb import FastTextEmbedding
from softmatch_db.tokenizers.mecab_tok import MeCabTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_PATH = BASE_DIR / "data" / "wiki_ja.txt"
EMBEDDING_PATH = BASE_DIR / "data" / "fasttext_ja" / "embeddings.npy"
VOCAB_PATH = BASE_DIR / "data" / "fasttext_ja" / "vocab.json"
INDEX_PATH = BASE_DIR / "web" / "index.duckdb"


def main() -> None:
    """Build the SoftMatch-DB DuckDB index."""
    logger.info("Corpus:    %s", CORPUS_PATH)
    logger.info("Embedding: %s", EMBEDDING_PATH)
    logger.info("Vocab:     %s", VOCAB_PATH)
    logger.info("Output:    %s", INDEX_PATH)

    embedding = FastTextEmbedding.load(str(EMBEDDING_PATH))
    tokenizer = MeCabTokenizer.from_fasttext_vocab(str(VOCAB_PATH))
    logger.info("MeCab tokenizer loaded (vocab=%d)", tokenizer.vocab_size)

    con = duckdb.connect(str(INDEX_PATH))
    try:
        softmatch_db.register(con)
        softmatch_db.build(
            con,
            corpus_path=str(CORPUS_PATH),
            lang="ja",
            tokenizer=tokenizer,
            embedding=embedding,
        )
        logger.info("Index build complete: %s", INDEX_PATH)
    finally:
        con.close()


if __name__ == "__main__":
    main()
