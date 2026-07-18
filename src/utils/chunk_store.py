import json
from pathlib import Path

CHUNK_FILE = Path("data/chunks.json")

def save_chunks(chunks):
    """
    Save all chunks to disk so BM25 can build a keyword index later.
    `chunks` are LangChain Document objects (from split_documents). 
    """

    records = []

    for chunk in chunks:

        records.append(
            {
                "text": chunk.page_content,
                "page": chunk.metadata.get("page", 0),
                "source": chunk.metadata.get("source", ""),
            }
        )
    CHUNK_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(CHUNK_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

        return records

def load_chunks():
    """
    Load save chunks for BM25.
    """

    if not CHUNK_FILE.exists():
        return []
    
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)