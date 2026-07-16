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
from src.retriever import retrieve
from src.retriever import build_context

from src.llm import generate_answer

def ask_question(question: str):
    results = retrieve(question)
    context = build_context(results)
    answer = generate_answer(
        context=context, 
        question=question
    )
    return answer
