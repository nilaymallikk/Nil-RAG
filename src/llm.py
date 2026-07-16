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
from src.config import CHAT_MODEL

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