from dotenv import load_dotenv
import os

load_dotenv()

# ==========================
# API KEYS
# ==========================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

# ==========================
# API URL
# ==========================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# ==========================
# MODELS
# ==========================

CHAT_MODEL: str = "nvidia/nemotron-3-super-120b-a12b:free"
EMBEDDING_MODEL = "jina-embeddings-v5-text-small"

# ==========================
# PINECONE
# ==========================

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# ==========================
# RAG PARAMETERS
# ==========================

CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200
TOP_K: int = 7
MIN_SCORE: float = 0.5
RRF_K: int = 60
# ==========================
# LLM SETTINGS
# ==========================

TEMPERATURE: float = 0.2

# ==========================
# Reranker Model
# ==========================

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K = 5
# ==========================
# Redis
# ==========================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CONVERSATION_TTL_SECONDS = int(
    os.getenv("CONVERSATION_TTL_SECONDS", "86400")
)