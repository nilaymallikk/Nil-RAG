import json
import hashlib
from pathlib import Path
from typing import Any

CHUNK_FILE = Path("data/chunks.json")


def build_chunk_id(*, text: str, source: str, page: int) -> str:
    """Return a deterministic identifier for one logical document chunk.

    The same source, page, and content always receive the same identifier.
    That lets the lexical and vector indexes refer to one shared chunk.
    """

    identity = f"{source}\n{page}\n{text}".encode("utf-8")
    return hashlib.sha256(identity).hexdigest()


def _to_record(chunk: Any) -> dict[str, Any]:
    """Create the canonical, serializable representation of a chunk."""

    text = chunk.page_content
    source = str(chunk.metadata.get("source", ""))
    page = int(chunk.metadata.get("page", 0))
    chunk_id = build_chunk_id(text=text, source=source, page=page)

    # Keep the in-memory Document aligned with its persisted representation.
    # Pinecone receives this identifier in the same indexing run.
    chunk.metadata["chunk_id"] = chunk_id

    return {
        "chunk_id": chunk_id,
        "text": text,
        "page": page,
        "source": source,
        "document": Path(source).stem,
    }


def save_chunks(chunks: list[Any]) -> list[dict[str, Any]]:
    """
    Save all chunks to disk so BM25 can build a keyword index later.
    `chunks` are LangChain Document objects (from split_documents). 
    """

    records = [_to_record(chunk) for chunk in chunks]
    CHUNK_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(CHUNK_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return records


def load_chunks() -> list[dict[str, Any]]:
    """
    Load save chunks for BM25.
    """

    if not CHUNK_FILE.exists():
        return []
    
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
