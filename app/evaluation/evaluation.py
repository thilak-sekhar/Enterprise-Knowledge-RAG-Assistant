from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    AnswerRelevancy
)
from ragas.llms import LangchainLLMWrapper
from langchain_community.chat_models import ChatOllama
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings


# -------- LLM (Evaluator) --------
llm = LangchainLLMWrapper(
    ChatOllama(
        model="llama3.1:8b",
        temperature=0
    )
)

# -------- Embeddings --------
embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
)


def evaluate_response(question: str, answer: str, documents: list) -> dict:

    # Extract raw chunk texts from retriever documents
    retrieved_chunks = [doc.page_content for doc in documents]

    if not retrieved_chunks or not answer.strip():
        return {
            "faithfulness": None,
            "context_precision": None,
            "context_recall": None,
            "answer_relevancy": None
        }

    data = {
        "question": [question],
        "answer": [answer],
        "retrieved_contexts": [retrieved_chunks]
    }

    dataset = Dataset.from_dict(data)

    metrics = [
        Faithfulness(llm=llm),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
        AnswerRelevancy(llm=llm)
    ]

    result = evaluate(
        dataset,
        metrics=metrics,
        embeddings=embeddings
    )

    return result
