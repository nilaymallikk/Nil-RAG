from src.vector_store import index
from src.embedding import get_embedding
from src.config import TOP_K, MIN_SCORE
from src.llm import generate_search_queries
from src.bm25 import bm25_search
from dataclasses import dataclass


# Retrive most relevent chunk

def retrieve(query: str):
    query_embedding = get_embedding(query)

    response = index.query(
        vector=query_embedding,
        top_k=TOP_K,
        include_metadata=True,
    )

    results = []

    for match in response.matches:

        if match.score < MIN_SCORE:
            continue

        results.append(
            {
                # Match IDs preserve compatibility with vectors created before
                # the shared chunk_id metadata contract was introduced.
                "chunk_id": match.metadata.get("chunk_id", match.id),
                "text": match.metadata["text"],
                "page": match.metadata.get("page", 0),
                "source": match.metadata.get("source", ""),
                "score": match.score,
            }
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
def filter_results(matches, min_score=MIN_SCORE):
    filtered = []
    for match in matches:
        if match["score"] >= min_score:
            filtered.append(match)
    return filtered

def extract_sources(matches):
    sources = []

    for match in matches:
        sources.append({
            "page": match["page"],
            "source": match["source"],
            "score": round(match["score"], 3),
        })

    return sources

# Function for Multi-Query Retrieval (MQR)

def retrieve_multi(question: str):

    # Generate search queries
    queries = generate_search_queries(question)

    #Include the original question
    queries.insert(0, question)

    # Search each query
    all_matches = []

    for query in queries:
        matches = hybrid_retrieve(query) # was: retrive(query)
        all_matches.extend(matches)
    
    # remove duplicate

    unique = {}

    for match in all_matches:

        key = (
            match["source"],
            match["page"],
            match["text"],
        )
        if key not in unique:
            unique[key] = match
        elif match["score"] > unique[key]["score"]:
            unique[key] = match 

    matches = list(unique.values())

    # RERANKER
    from src.reranker import rerank_matches

    matches = list(unique.values())
    matches.sort(key=lambda x: x["score"], reverse=True)
    matches = rerank_matches(question, matches)
    return matches[:TOP_K]

    # sort 
    matches.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Return only the best
    
    return matches[:TOP_K]

@dataclass(frozen=True)
class HybridCandidates:
    """Separate ranked candidates produced by each retrieval channel."""

    vector_matches: list[dict]
    bm25_matches: list[dict]


def retrieve_hybrid_candidates(query: str) -> HybridCandidates:
    """Retrieve dense and lexical candidates without comparing raw scores."""

    return HybridCandidates(
        vector_matches=retrieve(query),
        bm25_matches=bm25_search(query, top_k=TOP_K),
    )


def merge_hybrid_candidates(candidates: HybridCandidates) -> list[dict]:
    """Create a de-duplicated candidate set for the cross-encoder reranker.

    This is intentionally a union, not score fusion. Vector and BM25 scores
    are incomparable; RRF will introduce a principled ordering in the next
    roadmap step.
    """

    merged: dict[str, dict] = {}
    for channel, matches in (
        ("vector", candidates.vector_matches),
        ("bm25", candidates.bm25_matches),
    ):
        for match in matches:
            chunk_id = match["chunk_id"]
            if chunk_id not in merged:
                merged[chunk_id] = {**match, "retrieval_channels": [channel]}
            else:
                merged[chunk_id]["retrieval_channels"].append(channel)

    return list(merged.values())


def hybrid_retrieve(query: str):
    """
    Run vector and BM25 retrieval and return their candidate union.

    The downstream CrossEncoder determines the final ordering until RRF is
    added in the next roadmap step.
    """
    candidates = retrieve_hybrid_candidates(query)
    return merge_hybrid_candidates(candidates)
