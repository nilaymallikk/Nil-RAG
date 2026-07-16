from src.client import client
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

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content
