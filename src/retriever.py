from src.vector_store import index
from src.embedding import get_embedding
from src.config import TOP_K, MIN_SCORE
from src.llm import generate_search_queries


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
def filter_results(results, min_score=MIN_SCORE):
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
            "score": round(match.score, 3),
        }
        )
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
        matches = retrieve(query)
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

    # sort 
    matches.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Return only the best
    
    return matches[:TOP_K]



        


           



 