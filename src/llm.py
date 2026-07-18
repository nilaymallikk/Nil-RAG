from src.client import chat_client
from src.config import CHAT_MODEL
from src.utils.prompt_loader import load_prompt

"""
llm.py

Takes:
- User question
- Retrieved context

Returns:
- Generated answer from the LLM
"""


def generate_answer(context: str, question: str):

    template = load_prompt("rag_prompt.txt")

    prompt = template.format(
        context=context,
        question=question,
    )

    response = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content


# Add a helper for Multi-Query Retrieval (MQR)

from pathlib import Path
QUERY_PROMPT = Path("prompts/query_rewrite.txt").read_text(encoding="utf-8")

def generate_search_queries(question: str) -> list[str]:
    """
    Generate multiple search queries from a user question.
    """

    prompt = QUERY_PROMPT.format(
        question=question,
    )

    response = chat_client.chat.completions.create(
        model = CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    queries = response.choices[0].message.content.splitlines()

    return [
        q.strip("-• ").strip()
        for q in queries
        if q.strip()
    ]