from src.loader import load_pdf
from src.splitter import split_documents
from src.embedding import get_embeddings
from src.vector_store import upload_batches


def index_pdf(pdf_path: str):
    """
    Complete indexing pipeline.
    """

    from src.logger import logger
    logger.info(f"Indexing PDF: {pdf_path}")
    documents = load_pdf(pdf_path)

    logger.info(f"Loaded {len(documents)} pages")

    logger.info("Splitting documents...")
    chunks = split_documents(documents)

    logger.info(f"Created {len(chunks)} chunks")

    logger.info("Generating embeddings...")

    texts = [chunk.page_content for chunk in chunks]

    embeddings = get_embeddings(texts)

    logger.info("Uploading to Pinecone...")

    upload_batches(chunks, embeddings)

    logger.info("Indexing completed!")

    logger.error("Embedding generation failed")
    logger.warning("No relevant documents found")