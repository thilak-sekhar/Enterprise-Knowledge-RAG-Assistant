from app.chains.rag_chain import run_rag
from app.utils.logger import get_logger

logger = get_logger(__name__)

def run_query(query: str):
    """
    System entrypoint.
    Used by CLI, UI, API, tests.
    """
    logger.info(f"Received query: {query}")
    response = run_rag(query)

    # Expected response format:
    # {
    #   "answer": "...",
    #   "sources": [...]
    # }
    return response


def cli_mode():
    while True:
        query = input("Enter query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

        result = run_query(query)

        print("\nAnswer:")
        print(result.get("answer", ""))

        print("\nSources:")
        for src in result.get("sources", []):
            print(src)
        print("-" * 50)
        print("Evaluation Metrics:\n")
        evaluation = result.get("evaluation", {})
        print(evaluation)


if __name__ == "__main__":
    cli_mode()
