from pinecone import Pinecone, ServerlessSpec
import os
import loadenv
 
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
EMBEDDING_DIMENSION = 768  # must match your embedder
METRIC = "cosine"


def reset_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = pc.list_indexes().names()

    if PINECONE_INDEX_NAME in existing_indexes:
        print(f"[INFO] Deleting index: {PINECONE_INDEX_NAME}")
        pc.delete_index(PINECONE_INDEX_NAME)

    print(f"[INFO] Creating index: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION
        )
    )

    print("[SUCCESS] Pinecone index reset completed.")


if __name__ == "__main__":
    reset_index()
