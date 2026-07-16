from openai import OpenAI
from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)

# Create a local OpenAI client (points at OpenRouter-compatible base_url)
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)

DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"


# def test_embedding():
#     try:
#         response = client.embeddings.create(
#             model=DEFAULT_EMBEDDING_MODEL,
#             input="Artificial Intelligence"
#         )
#     except Exception as exc:
#         raise RuntimeError(
#             f"Embedding request failed for model {DEFAULT_EMBEDDING_MODEL}: {exc}"
#         ) from exc

#     data = getattr(response, "data", None)
#     if not data:
#         raise ValueError(
#             "No embedding data received from the embeddings endpoint. "
#             f"Response value: {response}"
#         )

#     return response

def get_embedding(text: str,):
    """
    Get the embedding for a given text using the specified model.
    """
    try:
        response = client.embeddings.create(
            model=DEFAULT_EMBEDDING_MODEL,
            input=text,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Embedding request failed for model {DEFAULT_EMBEDDING_MODEL}: {exc}"
        ) from exc

    data = getattr(response, "data", None)
    if not data:
        raise ValueError(
            "No embedding data received from the embeddings endpoint. "
            f"Response value: {response}"
        )

    return response.data[0].embedding