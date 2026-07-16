from pinecone import Pinecone, index
from src.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

from src.embedding import get_embedding

def upload_document(chunk, chunk_id):

    vector = get_embedding(chunk.page_content)
    index.upsert(
        vectors=[
            {
                "id": chunk_id,
                "values": vector,
                "metadata": {
                    "page": chunk.metadata["page"],
                    "source": chunk.metadata["source"],
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
