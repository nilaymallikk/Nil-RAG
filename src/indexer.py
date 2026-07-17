from src.loader import load_pdf
from src.splitter import split_documents
from src.embedding import get_embeddings
from src.vector_store import upload_batches


def index_pdf(pdf_path: str):
    """
    Complete indexing pipeline.
    """

    print("Loading PDF...")
    documents = load_pdf(pdf_path)

    print(f"Loaded {len(documents)} pages")

    print("Splitting documents...")
    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    print("Generating embeddings...")

    texts = [chunk.page_content for chunk in chunks]

    embeddings = get_embeddings(texts)

    print("Uploading to Pinecone...")

    upload_batches(chunks, embeddings)

    print("Indexing completed!")