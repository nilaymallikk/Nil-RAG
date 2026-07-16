try:
    from pinecone import Pinecone, index
except Exception:
    Pinecone = None
    index = None

from src.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)

if Pinecone is not None:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
else:
    pc = None

from src.embedding import get_embedding

def upload_document(chunk, chunk_id):
    if index is None:
        raise RuntimeError(
            "The 'pinecone' package is not installed or failed to import. "
            "Install dependencies with: pip install -r requirements.txt"
        )

    vector = get_embedding(chunk.page_content)
    index.upsert(
        vectors=[
            {
                "id": chunk_id,
                "values": vector,
                "metadata": {
                    "page": chunk.metadata["page"],
                    "source": chunk.metadata["source"],
                    "text": chunk.page_content,
                },
            }
        ]
    ) 

def upload_chunks(chunks):
    for i, chunk in enumerate(chunks):
        upload_document(chunk,
                        chunk_id=f"chunk-{i}"
        )

        print(f"Uploaded chunk {i} ")    
