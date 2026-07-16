"""
llm.py

Takes:

Question

Retrieved chunks

↓

Calls Nemotron

↓

Returns answer
"""
from openai import OpenAI

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
)

from src.client import client

DEFAULT_CHAT_MODEL = "tencent/hy3:free"

def generate_answer(context: str, question: str):
    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not contained in the context,
respond with:
"I couldn't find that information in the document."

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model=DEFAULT_CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content