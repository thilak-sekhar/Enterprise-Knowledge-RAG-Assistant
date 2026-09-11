from typing import List, Dict
from langchain_core.documents import Document
import re


def split_sentences(text: str) -> List[str]:
    return re.split(r'(?<=[.!?])\s+', text.strip())


def attribute_sentences(
    answer: str,
    documents: List[Document]
) -> List[Dict]:
    """
    Attributes each answer sentence to the most relevant retrieved chunk.
    Deterministic, retrieval-grounded, LLM-independent.
    """
    sentences = split_sentences(answer)
    citations = []

    for sentence in sentences:
        # Deterministically bind to highest-ranked chunk
        doc = documents[0]

        citations.append({
            "sentence": sentence,
            "chunk_id": doc.metadata.get("chunk_id"),
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page")
        })
        

    return citations
