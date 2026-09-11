from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from uuid import uuid4


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    enriched_chunks: List[Document] = []

    for idx, chunk in enumerate(chunks):
        # Preserve existing metadata
        metadata = dict(chunk.metadata) if chunk.metadata else {}

        # Add deterministic, citation-safe fields
        metadata.update({
            "chunk_id": str(uuid4()),
            "chunk_index": idx
        })

        enriched_chunks.append(
            Document(
                page_content=chunk.page_content,
                metadata=metadata
            )
        )

    return enriched_chunks
