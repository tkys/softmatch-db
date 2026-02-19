"""Tokenizer protocol definition."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol that all tokenizer implementations must satisfy."""

    def tokenize(self, line: str) -> list[str]:
        """Split a line of text into surface-form tokens.

        Args:
            line: Input text.

        Returns:
            List of token strings.
        """
        ...

    def encode(self, tokens: list[str]) -> list[int]:
        """Convert surface-form tokens to integer ids.

        Args:
            tokens: List of token strings.

        Returns:
            List of integer token ids.
        """
        ...

    def decode(self, ids: list[int]) -> list[str]:
        """Convert integer ids back to surface-form tokens.

        Args:
            ids: List of token ids.

        Returns:
            List of token strings.
        """
        ...

    @property
    def vocab_size(self) -> int:
        """Return the number of tokens in the vocabulary."""
        ...

    @property
    def unk_id(self) -> int:
        """Return the id used for unknown / out-of-vocabulary tokens."""
        ...
