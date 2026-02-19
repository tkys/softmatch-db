"""Japanese tokenizer using SudachiPy morphological analyser."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from sudachipy import Dictionary, SplitMode


class SudachiTokenizer:
    """SudachiPy-based tokenizer using ``SplitMode.C`` (longest split).

    Attributes:
        _dict: SudachiPy dictionary instance.
        _token_to_id: Surface form → integer id mapping.
        _id_to_token: Integer id → surface form mapping.
        _unk_id: Id for unknown tokens.
    """

    _UNK_TOKEN = "<UNK>"

    def __init__(
        self,
        token_to_id: dict[str, int],
        id_to_token: dict[int, str],
        unk_id: int,
    ) -> None:
        self._dict = Dictionary()
        self._tokenizer_obj = self._dict.create()
        self._token_to_id = token_to_id
        self._id_to_token = id_to_token
        self._unk_id = unk_id

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build_from_corpus(
        cls,
        corpus_path: str,
        max_vocab: int = 50000,
    ) -> SudachiTokenizer:
        """Build vocabulary by scanning a Japanese text corpus.

        Args:
            corpus_path: Path to a plain-text corpus (one sentence per line).
            max_vocab: Maximum vocabulary size including ``<UNK>``.

        Returns:
            New ``SudachiTokenizer`` with learned vocabulary.
        """
        tmp_dict = Dictionary()
        tmp_tok = tmp_dict.create()
        counter: Counter[str] = Counter()

        with open(corpus_path, encoding="utf-8") as fh:
            for line in fh:
                morphs = tmp_tok.tokenize(line.strip(), SplitMode.C)
                for m in morphs:
                    counter[m.surface().lower()] += 1

        token_to_id: dict[str, int] = {cls._UNK_TOKEN: 0}
        id_to_token: dict[int, str] = {0: cls._UNK_TOKEN}

        for idx, (tok, _) in enumerate(counter.most_common(max_vocab - 1), start=1):
            token_to_id[tok] = idx
            id_to_token[idx] = tok

        return cls(token_to_id, id_to_token, unk_id=0)

    @classmethod
    def from_vocab_file(cls, vocab_path: str) -> SudachiTokenizer:
        """Load vocabulary from a JSON file.

        Args:
            vocab_path: Path to a JSON mapping ``{token: id}``.

        Returns:
            Restored ``SudachiTokenizer``.
        """
        with open(vocab_path, encoding="utf-8") as fh:
            token_to_id: dict[str, int] = json.load(fh)
        id_to_token = {v: k for k, v in token_to_id.items()}
        unk_id = token_to_id.get(cls._UNK_TOKEN, 0)
        return cls(token_to_id, id_to_token, unk_id)

    def save_vocab(self, vocab_path: str) -> None:
        """Persist vocabulary to a JSON file.

        Args:
            vocab_path: Destination path.
        """
        Path(vocab_path).parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as fh:
            json.dump(self._token_to_id, fh, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Tokenizer protocol
    # ------------------------------------------------------------------

    def tokenize(self, line: str) -> list[str]:
        """Morphological analysis with SplitMode.C.

        Args:
            line: Input Japanese text.

        Returns:
            List of lowercased surface forms.
        """
        morphs = self._tokenizer_obj.tokenize(line.strip(), SplitMode.C)
        return [m.surface().lower() for m in morphs]

    def encode(self, tokens: list[str]) -> list[int]:
        """Map surface forms to integer ids.

        Args:
            tokens: List of surface form strings.

        Returns:
            List of token ids.
        """
        return [self._token_to_id.get(t, self._unk_id) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        """Map integer ids back to surface forms.

        Args:
            ids: List of token ids.

        Returns:
            List of surface form strings.
        """
        return [self._id_to_token.get(i, self._UNK_TOKEN) for i in ids]

    @property
    def vocab_size(self) -> int:
        """Number of tokens in vocabulary."""
        return len(self._token_to_id)

    @property
    def unk_id(self) -> int:
        """Unknown token id."""
        return self._unk_id

    def get_frequencies(self, corpus_path: str) -> list[int]:
        """Count token frequencies in a corpus file.

        Args:
            corpus_path: Path to plain-text corpus.

        Returns:
            List of length ``vocab_size`` with per-token counts.
        """
        freq = [0] * self.vocab_size
        with open(corpus_path, encoding="utf-8") as fh:
            for line in fh:
                for tok in self.tokenize(line):
                    tid = self._token_to_id.get(tok, self._unk_id)
                    freq[tid] += 1
        return freq
