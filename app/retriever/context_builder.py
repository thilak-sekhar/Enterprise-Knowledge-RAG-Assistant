def build_context(documents):
    """
    Builds a clean, readable context string from retrieved documents.
    """
    context_blocks = []

    for idx, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")

        block = (
            f"[Source {idx} | Page {page} | {source}]\n"
            f"{doc.page_content.strip()}"
        )
        context_blocks.append(block)

    return "\n\n".join(context_blocks)
