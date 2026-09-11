import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# Pinecone Configuration
# =========================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")

# =========================
# Ollama Configuration
# =========================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3.1:8b")

# =========================
# Embeddings
# =========================
EMBEDDING_MODEL_NAME = "nomic-embed-text"
EMBEDDING_DIMENSION = 768

# =========================
# Retrieval
# =========================
TOP_K = int(os.getenv("TOP_K", 5))

# =========================
# Validation (Fail Fast)
# =========================
if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise EnvironmentError(
        "Missing Pinecone configuration. Check your .env file."
    )

MIN_RETRIEVAL_SCORE = 0.25
MIN_CONTEXT_LENGTH = 300