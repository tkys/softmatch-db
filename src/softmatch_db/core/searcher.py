"""Searcher — integrates all components for end-to-end soft matching.

Loads data from DuckDB into in-memory structures and runs beam search.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from softmatch_db.core.beam_search import SearchResult, beam_search
from softmatch_db.core.ngram_filter import NgramBitFilter
from softmatch_db.core.softmin import softmin
from softmatch_db.core.sorted_index import SortedIndex

if TYPE_CHECKING:
    import duckdb

NDArrayF32 = NDArray[np.float32]


class Searcher:
    """In-memory searcher loaded from a DuckDB database.

    All data is loaded once at initialisation; subsequent searches are
    pure in-memory operations with no SQL round-trips.

    Attributes:
        embeddings: Normalised embedding matrix ``(V, D)``.
        norm_sq: Zipfian-whitened squared norms ``(V,)``.
        ngram_filter: N-gram bitset filter.
        sorted_index: Sorted hash index.
        token_to_id: Surface form → token id mapping.
        id_to_token: Token id → surface form mapping.
        lang: Language code (``"ja"`` or ``"en"``).
        _tokenize_fn: Tokenization function for query strings.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        """Load all index data from DuckDB into memory.

        Args:
            con: Open DuckDB connection with populated softmatch tables.

        Raises:
            RuntimeError: If required tables are empty or missing.
        """
        # 1. Vocabulary: embeddings, norms, token dictionary.
        vocab_df = con.sql(
            "SELECT token_id, token, freq, norm_sq, embedding "
            "FROM softmatch_vocab ORDER BY token_id"
        ).fetchdf()

        if vocab_df.empty:
            raise RuntimeError("softmatch_vocab table is empty")

        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        emb_list: list[NDArrayF32] = []

        for _, row in vocab_df.iterrows():
            tid = int(row["token_id"])
            tok = str(row["token"])
            self.token_to_id[tok] = tid
            self.id_to_token[tid] = tok
            emb_list.append(np.array(row["embedding"], dtype=np.float32))

        self.embeddings: NDArrayF32 = np.stack(emb_list)
        self.norm_sq: NDArrayF32 = np.array(
            vocab_df["norm_sq"].values, dtype=np.float32
        )

        # Pre-normalise vocabulary embeddings for cosine similarity.
        v_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        v_norms = np.maximum(v_norms, 1e-12)
        self._vocab_normed: NDArrayF32 = (self.embeddings / v_norms).astype(np.float32)

        # 2. N-gram filter from BLOBs.
        pair_row = con.sql(
            "SELECT cons, bits FROM softmatch_ngram_index "
            "WHERE ngram_type = 'pair'"
        ).fetchone()
        trio_row = con.sql(
            "SELECT cons, bits FROM softmatch_ngram_index "
            "WHERE ngram_type = 'trio'"
        ).fetchone()

        if pair_row is None or trio_row is None:
            raise RuntimeError("softmatch_ngram_index is incomplete")

        self.ngram_filter = NgramBitFilter.from_bytes(
            pair_cons=int(pair_row[0]),
            trio_cons=int(trio_row[0]),
            pair_blob=bytes(pair_row[1]),
            trio_blob=bytes(trio_row[1]),
        )

        # 3. Corpus tokens → SortedIndex.
        corpus_rows = con.sql(
            "SELECT tokens FROM softmatch_corpus ORDER BY doc_id, sent_id"
        ).fetchall()

        all_tokens: list[int] = []
        for (tok_list,) in corpus_rows:
            all_tokens.extend(int(t) for t in tok_list)

        corpus_arr = np.array(all_tokens, dtype=np.uint32)
        self.sorted_index = SortedIndex.build(corpus_arr)

        # 4. Config and tokenizer.
        lang_row = con.sql(
            "SELECT value FROM softmatch_config WHERE key = 'lang'"
        ).fetchone()
        self.lang: str = str(lang_row[0]) if lang_row else "ja"

        # Build a tokenizer for query-time use.
        self._tokenize_fn = self._build_tokenize_fn(self.lang)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _build_tokenize_fn(self, lang: str):
        """Create a tokenization function appropriate for the language.

        Uses MeCab/ipadic for Japanese (matching official SoftMatcha2),
        with SudachiPy as fallback.

        Args:
            lang: Language code (``"ja"`` or ``"en"``).

        Returns:
            A callable ``str -> list[str]``.
        """
        if lang == "ja":
            try:
                import ipadic
                import MeCab

                _tagger = MeCab.Tagger(f"-Owakati {ipadic.MECAB_ARGS}")

                def _tokenize_ja(text: str) -> list[str]:
                    parsed = _tagger.parse(text.strip())
                    if parsed is None:
                        return []
                    return [tok for tok in parsed.rstrip().split(" ") if tok]

                return _tokenize_ja
            except ImportError:
                pass
            try:
                from sudachipy import Dictionary, SplitMode

                _dict = Dictionary()
                _tok = _dict.create()

                def _tokenize_ja_sudachi(text: str) -> list[str]:
                    morphs = _tok.tokenize(text.strip(), SplitMode.C)
                    return [m.surface().lower() for m in morphs]

                return _tokenize_ja_sudachi
            except ImportError:
                pass
        # Fallback: whitespace split.
        return lambda text: text.strip().lower().split()

    def _tokenize_and_encode(self, query: str) -> list[int]:
        """Tokenize a query string and map to token ids.

        Args:
            query: Query string.

        Returns:
            List of token ids.
        """
        tokens = self._tokenize_fn(query)
        return [self.token_to_id.get(t, 0) for t in tokens]

    def search(
        self,
        query: str,
        top_k: int = 20,
        min_similarity: float = 0.45,
        max_runtime: float = 10.0,
    ) -> list[SearchResult]:
        """Run a soft-matching search for the given query.

        The query is tokenised, and a cosine similarity matrix is
        computed against the full vocabulary. Beam search then finds
        corpus-attested patterns similar to the query.

        Args:
            query: Query string.
            top_k: Maximum number of results.
            min_similarity: Minimum acceptable score.
            max_runtime: Time budget in seconds.

        Returns:
            List of ``SearchResult`` sorted by descending score.
        """
        ids = self._tokenize_and_encode(query)

        if not ids:
            return []

        # Compute score matrix: (pat_len, V).
        score_matrix = self._compute_similarity(ids)

        return beam_search(
            pattern_tokens=ids,
            score_matrix=score_matrix,
            norm_sq=self.norm_sq,
            ngram_filter=self.ngram_filter,
            sorted_index=self.sorted_index,
            top_k=top_k,
            min_similarity=min_similarity,
            max_runtime=max_runtime,
        )

    def score(self, query: str, tokens: list[int]) -> float:
        """Compute the softmin score between *query* and a token sequence.

        Intended for use as a scalar UDF.

        Args:
            query: Query string.
            tokens: Candidate token id sequence.

        Returns:
            Similarity score in [0, 1].
        """
        q_ids = self._tokenize_and_encode(query)
        if not q_ids or not tokens:
            return 0.0

        score_mat = self._compute_similarity(q_ids)

        # Greedy softmin across aligned positions.
        sim = 1.0
        for i, _qid in enumerate(q_ids):
            if i >= len(tokens):
                break
            v = tokens[i]
            if v < score_mat.shape[1]:
                sim = softmin(sim, float(score_mat[i, v]))
        return sim

    # ------------------------------------------------------------------
    # KWIC (Key Word In Context)
    # ------------------------------------------------------------------

    def find_examples(
        self,
        con: duckdb.DuckDBPyConnection,
        token_ids: list[int],
        max_examples: int = 5,
    ) -> list[dict[str, int | str | list[int]]]:
        """Find corpus sentences containing the given token subsequence.

        Searches ``softmatch_corpus`` for sentences whose ``tokens`` array
        contains *token_ids* as a contiguous subsequence, then returns the
        matching sentences with highlight positions.

        Args:
            con: Open DuckDB connection.
            token_ids: Token id sequence to search for.
            max_examples: Maximum number of example sentences to return.

        Returns:
            List of dicts with keys ``text``, ``sent_id``, ``tokens``,
            ``match_start``, ``match_end`` (character offsets in *text*
            for ``<mark>`` highlighting).
        """
        if not token_ids:
            return []

        pat_len = len(token_ids)

        # Build a SQL filter: tokens must contain ALL ids in the pattern.
        # This is a fast pre-filter; we verify contiguous subsequence in Python.
        contains_clauses = " AND ".join(
            f"list_contains(tokens, {tid})" for tid in token_ids
        )

        rows = con.sql(
            f"SELECT doc_id, sent_id, text, tokens "
            f"FROM softmatch_corpus "
            f"WHERE {contains_clauses} "
            f"LIMIT {max_examples * 5}"
        ).fetchall()

        results: list[dict[str, int | str | list[int]]] = []

        for doc_id, sent_id, text, tokens in rows:
            tok_list = [int(t) for t in tokens]
            # Find contiguous subsequence match.
            match_pos = self._find_subsequence(tok_list, token_ids)
            if match_pos < 0:
                continue

            # Compute character offsets for highlighting.
            # Concatenate matched token surfaces and find them in the
            # original text.  The corpus text is the raw sentence (no
            # spaces for Japanese), so we search for the joined surface.
            match_surfaces = [
                self.id_to_token.get(t, "") for t in tok_list[match_pos : match_pos + pat_len]
            ]
            match_str = "".join(match_surfaces)
            char_start = str(text).find(match_str)
            if char_start < 0:
                # Fallback: try with spaces (English-style text).
                match_str = " ".join(match_surfaces)
                char_start = str(text).find(match_str)
            if char_start < 0:
                # Cannot locate match in text; skip this example.
                continue
            match_len = len(match_str)

            results.append({
                "text": str(text),
                "sent_id": int(sent_id),
                "tokens": tok_list,
                "match_start": char_start,
                "match_end": char_start + match_len,
            })

            if len(results) >= max_examples:
                break

        return results

    @staticmethod
    def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
        """Find the first occurrence of *needle* in *haystack*.

        Args:
            haystack: Token id sequence to search in.
            needle: Token id subsequence to find.

        Returns:
            Start index of the first match, or ``-1`` if not found.
        """
        n_len = len(needle)
        for i in range(len(haystack) - n_len + 1):
            if haystack[i : i + n_len] == needle:
                return i
        return -1

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_similarity(self, query_ids: list[int]) -> NDArrayF32:
        """Build cosine similarity matrix ``(pat_len, V)``.

        Args:
            query_ids: Token ids of the query.

        Returns:
            Score matrix of shape ``(len(query_ids), V)``.
        """
        q_emb = self.embeddings[query_ids]  # (P, D)
        # Normalise query embeddings.
        q_norms = np.linalg.norm(q_emb, axis=1, keepdims=True)
        q_norms = np.maximum(q_norms, 1e-12)
        q_emb_n = q_emb / q_norms

        # Use pre-normalised vocabulary embeddings.
        scores = (q_emb_n @ self._vocab_normed.T).astype(np.float32)
        return scores

    def decode_results(self, results: list[SearchResult]) -> pd.DataFrame:
        """Convert search results to a DataFrame with decoded tokens.

        Args:
            results: List of ``SearchResult``.

        Returns:
            DataFrame with columns ``rank``, ``tokens``, ``text``,
            ``score``, ``count``.
        """
        rows = []
        for rank, r in enumerate(results, start=1):
            text = " ".join(self.id_to_token.get(t, "<UNK>") for t in r.tokens)
            rows.append({
                "rank": rank,
                "tokens": r.tokens,
                "text": text,
                "score": r.score,
                "count": r.count,
            })
        return pd.DataFrame(rows)
