from src.vector_store import index
from src.embedding import get_embedding
from src.config import TOP_K


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

def build_context(matches):
    context = []
    for match in matches:
        context.append(match.metadata["text"])
    return "\n\n".join(context)

MIN_SCORE = 0.4
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