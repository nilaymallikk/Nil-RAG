from sentence_transformers import CrossEncoder

from src.config import RERANKER_MODEL, RERANK_TOP_K

reranker = CrossEncoder(RERANKER_MODEL)

def rerank_matches(question: str, matches: list[dict]) -> list [dict]:
    if not matches:
        return []
    

    pairs = [(question, match["text"]) for match in matches]
    scores = reranker.predict(pairs)
    
    reranked = []
    for match, score in zip(matches, scores):
        iteam = match.copy()
        iteam["rerank_score"] = float(score)
        reranked.append(iteam)
    
    reranked.sort(key=lambda x:x["rerank_score"], reverse= True)
    return reranked[:RERANK_TOP_K]