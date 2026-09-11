from langchain_pinecone import PineconeVectorStore
from app.embeddings.embedder import get_embeddings
from app.config.settings import PINECONE_INDEX_NAME, TOP_K
from app.utils.logger import get_logger

logger = get_logger(__name__)

def get_retriever():
    """
    Returns a LangChain retriever object (NOT documents).
    """
    embeddings = get_embeddings()

    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": TOP_K,
            "score_threshold": 0.65
        }
    )
    logger.info("Created Pinecone retriever.")
    return retriever


