try:
    from openai import OpenAI
except Exception:  # ImportError or any other import-time issue
    OpenAI = None

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)

# Create a local OpenAI client (points at OpenRouter-compatible base_url)
if OpenAI is not None:
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
else:
    client = None

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
    if client is None:
        raise RuntimeError(
            "The 'openai' package is not installed or failed to import. "
            "Install dependencies with: pip install -r requirements.txt"
        )

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