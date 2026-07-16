"""
retriever.py

User asks a question.

↓

Convert question into embedding

↓

Search Pinecone

↓

Return the most relevant chunks
"""

from src.vector_store import index
from src.embedding import get_embedding


def retrieve(query: str, top_k: int = 5):
    """
    Retrieve the most relevant chunks for a user query.
    """

    query_embedding = get_embedding(query)

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    return results

def build_context(results):
    context = []
    for match in results.matches:
        context.append(match.metadata["text"])
    return "\n\n".join(context)
