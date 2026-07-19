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


# def generate_answer(context: str, question: str, memory=None):

#     template = load_prompt("rag_prompt.txt")

#     prompt = template.format(
#         context=context,
#         question=question,
#     )

#     # print(f"\n[DEBUG] Context preview:\n{prompt[:800]}")

#     history = memory.get_messages() if memory else []
#     # History first, then current question.
#     # The LLM reads top-to-bottom so it sees the full conversation
#     # before generating the response.
#     print(f"\n[DEBUG] Memory has {len(history)} messages")

#     messages = history + [{"role":"user", "content": prompt}]

#     # memory.get_messages() if memory else [] — this is a ternary. If no memory was passed (old callers), history is empty and it behaves exactly as before. No breaking change.

#      #  history + [...] — Python list concatenation. History goes first because the LLM reads the messages in order. The current question (wrapped in the full RAG prompt with context) comes last.

#     response = chat_client.chat.completions.create(
#         model=CHAT_MODEL,
#         messages=messages,
#     )

#     return response.choices[0].message.content


history = memory.get_messages() if memory else []

current_message = {
    "role": "user",
    "content": prompt,
}

messages = [*history, current_message]

for position, message in enumerate(messages):
    if not isinstance(message, dict):
        raise TypeError(
            f"Invalid chat message at position {position}: "
            f"expected dict, got {type(message).__name__}."
        )

    if message.get("role") not in {"user", "assistant", "system"}:
        raise ValueError(
            f"Invalid chat role at position {position}: {message.get('role')!r}"
        )

    if not isinstance(message.get("content"), str):
        raise TypeError(
            f"Invalid message content at position {position}: "
            "expected a string."
        )

print(f"\n[DEBUG] Memory has {len(history)} messages")

response = chat_client.chat.completions.create(
    model=CHAT_MODEL,
    messages=messages,
)

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