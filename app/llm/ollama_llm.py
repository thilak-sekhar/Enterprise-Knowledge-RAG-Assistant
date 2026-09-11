from app.config.settings import OLLAMA_MODEL_NAME
from app.utils.logger import get_logger
from langchain_ollama import OllamaLLM


logger = get_logger(__name__)

def get_llm():
    """
    Returns an Ollama LLM instance.
    """
    logger.info(f"Using Ollama model: {OLLAMA_MODEL_NAME}")

    return OllamaLLM(
        model="llama3.1:8b",
        temperature=0.2,
        )

