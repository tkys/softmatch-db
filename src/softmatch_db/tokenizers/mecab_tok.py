"""Japanese tokenizer using MeCab with ipadic dictionary.

Matches the official SoftMatcha2 tokenizer (``TokenizerMecab``).
Uses ``MeCab.Tagger`` with ``-Owakati`` (space-delimited output)
and the ipadic dictionary.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import ipadic
import MeCab


class MeCabTokenizer:
    """MeCab/ipadic-based tokenizer matching official SoftMatcha2.

    Attributes:
        _tagger: MeCab Tagger instance.
        _token_to_id: Surface form → integer id mapping.
        _id_to_token: Integer id → surface form mapping.
        _unk_id: Id for unknown tokens.
    """

    _UNK_TOKEN = "<unk>"

    def __init__(
        self,
        token_to_id: dict[str, int],
        id_to_token: dict[int, str],
        unk_id: int,
    ) -> None:
        self._tagger = MeCab.Tagger(f"-Owakati {ipadic.MECAB_ARGS}")
        self._token_to_id = token_to_id
        self._id_to_token = id_to_token
        self._unk_id = unk_id

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_fasttext_vocab(cls, vocab_path: str) -> MeCabTokenizer:
        """Build tokenizer from a FastText vocab.json file.

        The vocab.json maps surface forms to integer ids and is derived
        directly from the ``.vec`` file header, matching the official
        SoftMatcha2 vocabulary construction.

        Args:
            vocab_path: Path to ``vocab.json`` (``{token: id}``).

        Returns:
            New ``MeCabTokenizer`` with FastText vocabulary.
        """
        with open(vocab_path, encoding="utf-8") as fh:
            token_to_id: dict[str, int] = json.load(fh)
        if cls._UNK_TOKEN not in token_to_id:
            unk_id = max(token_to_id.values()) + 1
            token_to_id[cls._UNK_TOKEN] = unk_id
        else:
            unk_id = token_to_id[cls._UNK_TOKEN]
        id_to_token = {v: k for k, v in token_to_id.items()}
        return cls(token_to_id, id_to_token, unk_id)

    @classmethod
    def build_from_corpus(
        cls,
        corpus_path: str,
        max_vocab: int = 50000,
    ) -> MeCabTokenizer:
        """Build vocabulary by scanning a Japanese text corpus.

        Args:
            corpus_path: Path to a plain-text corpus (one sentence per line).
            max_vocab: Maximum vocabulary size including ``<unk>``.

        Returns:
            New ``MeCabTokenizer`` with learned vocabulary.
        """
        tagger = MeCab.Tagger(f"-Owakati {ipadic.MECAB_ARGS}")
        counter: Counter[str] = Counter()

        with open(corpus_path, encoding="utf-8") as fh:
            for line in fh:
                parsed = tagger.parse(line.strip())
                if parsed:
                    for tok in parsed.rstrip().split(" "):
                        if tok:
                            counter[tok] += 1

        token_to_id: dict[str, int] = {cls._UNK_TOKEN: 0}
        id_to_token: dict[int, str] = {0: cls._UNK_TOKEN}

        for idx, (tok, _) in enumerate(counter.most_common(max_vocab - 1), start=1):
            token_to_id[tok] = idx
            id_to_token[idx] = tok

        return cls(token_to_id, id_to_token, unk_id=0)

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
        """Morphological analysis with MeCab/ipadic ``-Owakati``.

        Args:
            line: Input Japanese text.

        Returns:
            List of surface forms (as-is, no lowering — MeCab tokens
            are already lowered where appropriate).
        """
        parsed = self._tagger.parse(line.strip())
        if parsed is None:
            return []
        return [tok for tok in parsed.rstrip().split(" ") if tok]

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
