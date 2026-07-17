from src.client import embedding_client
from src.config import EMBEDDING_MODEL


def get_embedding(text: str):
    response = embedding_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def get_embeddings(texts: list[str]):
    response = embedding_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]