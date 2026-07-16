from openai import OpenAI
from google import genai
from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
)

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)

genai.configure(api_key=GEMINI_API_KEY)
genai_client = genai.GenerativeModel(GEMINI_BASE_URL)