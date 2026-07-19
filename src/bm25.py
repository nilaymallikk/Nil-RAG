"""Lexical retrieval over the locally persisted chunk catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import warnings

from rank_bm25 import BM25Okapi

from src.utils.chunk_store import CHUNK_FILE, build_chunk_id, load_chunks


def tokenize(text: str) -> list[str]:
    """Normalize text into the tokens used by the lexical retriever.

    This deliberately remains simple for now. Tokenization is isolated here so
    it can later be upgraded (for example, with stemming or language-aware
    analyzers) without changing the retrieval contract.
    """

    return text.lower().split()


@dataclass
class BM25Retriever:
    """A reloadable BM25 index backed by the canonical chunk catalog."""

    chunk_file: Path = CHUNK_FILE
    chunk_loader: Callable[[], list[dict[str, Any]]] = load_chunks
    _chunks: list[dict[str, Any]] = field(default_factory=list, init=False)
    _index: BM25Okapi | None = field(default=None, init=False)
    _catalog_mtime_ns: int | None = field(default=None, init=False)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return lexical candidates using the common retrieval result shape."""

        if top_k <= 0:
            return []

        self._refresh_if_needed()
        if self._index is None:
            return []

        scores = self._index.get_scores(tokenize(query))
        ranked = sorted(
            zip(self._chunks, scores, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_k]

        return [
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "page": chunk["page"],
                "source": chunk["source"],
                "score": float(score),
            }
            for chunk, score in ranked
        ]

    def _refresh_if_needed(self) -> None:
        """Rebuild only when the persisted catalog changed since last load."""

        mtime_ns = self._get_catalog_mtime_ns()
        if mtime_ns == self._catalog_mtime_ns:
            return

        chunks = self._normalize_chunks(self.chunk_loader())

        self._chunks = chunks
        self._index = (
            BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])
            if chunks
            else None
        )
        self._catalog_mtime_ns = mtime_ns

    def _get_catalog_mtime_ns(self) -> int | None:
        try:
            return self.chunk_file.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    @staticmethod
    def _normalize_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate catalog records and support pre-identity catalogs briefly."""

        required_fields = {"text", "page", "source"}
        normalized_chunks = []
        legacy_catalog = False

        for position, chunk in enumerate(chunks):
            missing_fields = required_fields - chunk.keys()
            if missing_fields:
                fields = ", ".join(sorted(missing_fields))
                raise ValueError(
                    f"Chunk catalog entry {position} is missing required fields: {fields}."
                )

            normalized_chunk = dict(chunk)
            if "chunk_id" not in normalized_chunk:
                legacy_catalog = True
                normalized_chunk["chunk_id"] = build_chunk_id(
                    text=str(normalized_chunk["text"]),
                    source=str(normalized_chunk["source"]),
                    page=int(normalized_chunk["page"]),
                )
            normalized_chunks.append(normalized_chunk)

        if legacy_catalog:
            warnings.warn(
                "Loaded a legacy chunk catalog without chunk_id values. "
                "Re-run indexing to persist the shared hybrid-search identity contract.",
                stacklevel=2,
            )

        return normalized_chunks


_default_retriever = BM25Retriever()


def bm25_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Compatibility entry point for lexical retrieval."""

    return _default_retriever.search(query=query, top_k=top_k)
