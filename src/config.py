from dotenv import load_dotenv
import os

load_dotenv()

# ==========================
# API KEYS
# ==========================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ==========================
# API URL
# ==========================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta2/models/chat-bison-001:generateMessage"

# ==========================
# MODELS
# ==========================

CHAT_MODEL: str = "tencent/hy3:free"
EMBEDDING_MODEL: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"

# ==========================
# PINECONE
# ==========================

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# ==========================
# RAG PARAMETERS
# ==========================

CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200
TOP_K: int = 10

# ==========================
# LLM SETTINGS
# ==========================

TEMPERATURE: float = 0.2