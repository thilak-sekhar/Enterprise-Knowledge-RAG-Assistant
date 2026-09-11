from app.retriever.retriever import get_retriever
from app.retriever.context_builder import build_context
from app.prompts.rag_prompt import QA_PROMPT
from app.llm.ollama_llm import get_llm
from app.utils.citations import attribute_sentences
from app.utils.logger import get_logger
from app.utils.refusal import is_refusal
from app.evaluation.evaluation import evaluate_response

logger = get_logger(__name__)


def run_rag(query: str) -> dict:
    retriever = get_retriever()
    documents = retriever.invoke(query)

    if not documents:
        return {
            "answer": None,
            "citations": [],
            "sources": [],
            "refused": True
        }
    logger.info(f"Retrieved documents for the query.")

    context = build_context(documents)
    logger.info("Built context from retrieved documents.")

    prompt = QA_PROMPT.format(
        context=context,
        question=query
    )
    logger.info("Constructed prompt for LLM.")


    llm = get_llm()
    answer = llm.invoke(prompt).strip()
    if is_refusal(answer):
        return {
            "answer": answer,
            "citations": [],
            "sources": [],
            "refused": True
        }
    logger.info("LLM generated an answer.")


    citations = attribute_sentences(answer, documents)
    logger.info("Attributed sentences to sources.")

    sources = list({
        doc.metadata.get("source") for doc in documents
    })
    logger.info("Compiled succesfully.")

    evaluation_metrics = evaluate_response(query, answer, documents)

    return {
        "answer": answer,
        "citations": citations,
        "sources": sources,
        "refused": False,
        "evaluation": evaluation_metrics
    }

if __name__ == "__main__":
    result = run_rag("What is glass box evaluation?")
    print(result["answer"])
    print("Sources:", result["sources"])
