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

def test_llm():

    response = client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        messages=[
            {
                "role":"user",
                "content":"Say hello in one sentence"
            }
        ]
    )

    return response