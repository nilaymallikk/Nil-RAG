from rank_bm25 import BM25Okapi

from src.utils.chunk_store import load_chunks

def _tokenize(text: str):
    """
    Simple tokenizer: lowercase + split on whiteplace
    """
    return text.lower().split()

# Build the index once, at import time.
_chunks = load_chunks()
_bm25 = None

if _chunks:
    _tokenize_corpus = [_tokenize(c["text"]) for c in _chunks]
    _bm25 = BM25Okapi(_tokenize_corpus)

def bm25_search(query: str, top_k: int =5):
    """
    Keyword search over the local chunk corpus.
    Returns the same dict shape as the vector retriever.
    """
    if _bm25 is None:
        return []
    
    scores = _bm25.get_scores(_tokenize(query))

    # Pair each chunk with its score, take the top_k highest.
    ranked = sorted(
        zip(_chunks, scores),
        key=lambda pair: pair[1],
        reverse=True,
    )[:top_k]

    results = []

    for chunk, score in ranked:
        results.append(
            {
                "text": chunk["text"],
                "page": chunk["page"],
                "source": chunk["source"],
                "score": float(score),
            }
        )
    return results

