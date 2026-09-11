from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List
import os

def load_pdf(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


def load_documents() -> List[Document]:
    """
    Loads all PDFs from the top-level 'pdf/' directory.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pdf_dir = os.path.join(project_root, "pdf")

    all_documents: List[Document] = []

    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_dir, filename)
            docs = load_pdf(file_path)
            all_documents.extend(docs)
            print(pdf_dir)
            print(os.listdir(pdf_dir))


    return all_documents



