"""
chatbot.py

Combines everything.

Question

↓

Retriever

↓

LLM

↓

Answer
"""
from src.retriever import (
    retrieve,
    filter_results,
    build_context,
    extract_sources,
    retrieve_multi
)

from src.llm import generate_answer

def ask_question(question: str):
    # step 1: retrive candidates 
    matches = retrieve_multi(question)

    # step 2: filter weak match
    matches = filter_results(matches)
    print("\nRetrieved Results:")
    for match in matches:
        print(
            f"Score: {match['score']:.3f} | "
            f"Page: {match['page']} | "
            f"Source: {match['source']}"
        )

    # step 3: handel no matches
    if not matches:
        return {
            "answer": "I couldn't find relevant information in the document.",
            "sources": [],
        }
    
    # step 4: build context
    context = build_context(matches)

    # step 5: generate answer
    answer = generate_answer(
        context=context, 
        question=question
    )
    #step 6: extract sources
    sources = extract_sources(matches)
    return {
        "answer": answer,
        "sources": sources
    }   