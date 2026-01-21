from typing import Iterable, Optional, Set
from naga.racy_lock import RacyLock


class KeywordSet:
    """
    A case-sensitive set of strings, for use with Namer to avoid collisions
    with keywords and other reserved identifiers.
    """

    def __init__(self, items: Optional[Iterable[str]] = None):
        self._set: Set[str] = set(items) if items else set()

    @staticmethod
    def new() -> "KeywordSet":
        """Returns a new mutable empty set."""
        return KeywordSet()

    @staticmethod
    def empty() -> "KeywordSet":
        """Returns a reference to the empty set."""
        if not hasattr(KeywordSet, "_EMPTY"):
            KeywordSet._EMPTY = RacyLock(KeywordSet)
        return KeywordSet._EMPTY.get()

    def contains(self, identifier: str) -> bool:
        """Returns whether the set contains the given string."""
        return identifier in self._set

    @classmethod
    def from_iter(cls, items: Iterable[str]) -> "KeywordSet":
        """Creates a KeywordSet from an iterator."""
        return cls(items)

    def extend(self, items: Iterable[str]) -> None:
        """Extends the set with items from an iterator."""
        self._set.update(items)


class CaseInsensitiveKeywordSet:
    """
    A case-insensitive, ASCII-only set of strings, for use with Namer to avoid
    collisions with keywords and other reserved identifiers.
    """

    def __init__(self, items: Optional[Iterable[str]] = None):
        self._set: Set[str] = set(s.lower() for s in items) if items else set()

    @staticmethod
    def new() -> "CaseInsensitiveKeywordSet":
        """Returns a new mutable empty set."""
        return CaseInsensitiveKeywordSet()

    @staticmethod
    def empty() -> "CaseInsensitiveKeywordSet":
        """Returns a reference to the empty set."""
        if not hasattr(CaseInsensitiveKeywordSet, "_EMPTY"):
            CaseInsensitiveKeywordSet._EMPTY = RacyLock(CaseInsensitiveKeywordSet)
        return CaseInsensitiveKeywordSet._EMPTY.get()

    def contains(self, identifier: str) -> bool:
        """
        Returns whether the set contains the given string, with case-insensitive
        comparison.
        """
        return identifier.lower() in self._set

    @classmethod
    def from_iter(cls, items: Iterable[str]) -> "CaseInsensitiveKeywordSet":
        """Creates a CaseInsensitiveKeywordSet from an iterator."""
        return cls(items)

    def extend(self, items: Iterable[str]) -> None:
        """Extends the set with items from an iterator."""
        self._set.update(s.lower() for s in items)
