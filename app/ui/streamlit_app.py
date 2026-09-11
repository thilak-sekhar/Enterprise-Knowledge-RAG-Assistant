import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
from app.main import run_query

# Page config
st.set_page_config(
    page_title="Enterprise Knowledge RAG Assistant",
    page_icon="📘",
    layout="wide"
)

# Header
st.markdown(
    """
    <h1 style="margin-bottom:0;">Enterprise Knowledge RAG Assistant</h1>
    <p style="color:gray; margin-top:4px;">
    LangChain • Pinecone • Retrieval-Augmented Generation
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# Sidebar
st.sidebar.header("System Info")
st.sidebar.markdown(
    """
    **Architecture**
    - LangChain RAG
    - Pinecone Vector DB
    - Incremental Ingestion
    - Refusal-safe responses

    **Usage**
    - Ask factual questions
    - Answers are grounded in indexed documents
    """
)

# Query input
st.subheader("Ask a Question")
query = st.text_input(
    "",
    placeholder="Enter a question related to the knowledge base..."
)

run = st.button("Run Query", type="primary")

if run and query:
    with st.spinner("Retrieving context and generating answer..."):
        result = run_query(query)

    # Answer section
    st.divider()
    st.subheader("Answer")
    st.write(result.get("answer", ""))

    # Sources
    sources = result.get("sources", [])
    if sources:
        st.divider()
        st.subheader("Retrieved Sources")

        for i, src in enumerate(sources, start=1):
            with st.expander(f"Source {i}"):
                st.write(src)
    else:
        st.info("No supporting sources returned.")
