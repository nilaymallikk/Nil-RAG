from langchain_community.document_loaders import PyPDFLoader

def load_pdf(path: str):
    """
    Load a PDF file and return its pages as a list of documents.
    """
    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents