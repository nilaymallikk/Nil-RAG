from pinecone import Pinecone
from pathlib import Path

from src.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)

# -----------------------------
# Initialize Pinecone
# -----------------------------

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


# -----------------------------
# Build Pinecone vectors
# -----------------------------

def build_vectors(chunks, embeddings):
    """
    Convert chunks + embeddings into Pinecone vector format.
    """

    vectors = []

    for chunk, embedding in zip(chunks, embeddings):

        vectors.append(
            {
                # A deterministic ID makes re-indexing idempotent and gives
                # BM25 and Pinecone a shared identity for this logical chunk.
                "id": chunk.metadata["chunk_id"],
                "values": embedding,
                "metadata": {
                    "chunk_id": chunk.metadata["chunk_id"],
                    "text": chunk.page_content,
                    "page": chunk.metadata.get("page", 0),
                    "source": chunk.metadata.get("source", ""),
                    "document": Path(
                        chunk.metadata.get("source", "")
                    ).stem,
                },
            }
        )

    return vectors


# -----------------------------
# Upload vectors
# -----------------------------

def upload_vectors(vectors):
    """
    Upload vectors to Pinecone.
    """

    index.upsert(vectors=vectors)


# -----------------------------
# Batch Upload
# -----------------------------

def upload_batches(chunks, embeddings, batch_size=100):
    """
    Upload vectors in batches.
    """

    total = len(chunks)

    for start in range(0, total, batch_size):

        end = start + batch_size

        chunk_batch = chunks[start:end]
        embedding_batch = embeddings[start:end]

        vectors = build_vectors(
            chunk_batch,
            embedding_batch,
        )

        upload_vectors(vectors)

        print(
            f"Uploaded {min(end, total)}/{total} chunks"
        )
