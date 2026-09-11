from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_documents
from app.embeddings.embedder import get_embeddings
from langchain_pinecone import PineconeVectorStore
from app.config.settings import TOP_K
from app.utils.logger import get_logger
from app.config.settings import PINECONE_INDEX_NAME

logger = get_logger(__name__)



def ingest_pipeline():
    logger.info("Starting ingestion pipeline")
    documents = load_documents()
    # 2. Chunk documents (with chunk_id, chunk_index)
    chunks = chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    # 3. Load embeddings
    embeddings = get_embeddings()
    # 4. Get vectorstore (Pinecone)
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,  # Replace with your actual index name
        embedding=embeddings
    )
    # 5. Index documents
    vectorstore.add_documents(chunks)
    logger.info("Documents successfully indexed into Pinecone")

if __name__ == "__main__":
    ingest_pipeline()