from src.vector_store import index
from src.embedding import get_embedding
from src.config import TOP_K, MIN_SCORE


def retrieve(query: str):
    """
    Retrieve the most relevant chunks for a user query.
    """

    query_embedding = get_embedding(query)

    results = index.query(
        vector=query_embedding,
        top_k=TOP_K,
        include_metadata=True,
    )

    return results

# Improve context building

def build_context(matches):
    """
    Convert retrieved matches into a prompt context.
    """

    passages = []

    for i, match in enumerate(matches, start=1):

        passages.append(
            f"""Passage {i}
Source: {match['source']}
Page: {match['page']}

{match['text']}"""
        )

    return "\n\n-------------------------\n\n".join(passages)

MIN_SCORE = MIN_SCORE  # Use the MIN_SCORE from config.py
def filter_results(results, min_score=MIN_SCORE):
    filtered = []
    for match in results.matches:
        if match.score >= min_score:
            filtered.append(match)
    return filtered

def extract_sources(matches):
    sources = []
    for match in matches:
        sources.append({
            "page": match.metadata["page"],
            "source": match.metadata["source"],
            "score": round(match.score, 3),
        }
        )
    return sources