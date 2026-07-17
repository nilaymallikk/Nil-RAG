from openai import OpenAI
from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    JINA_API_KEY,
)

chat_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)

embedding_client = OpenAI(
    api_key=JINA_API_KEY,
    base_url="https://api.jina.ai/v1",
)