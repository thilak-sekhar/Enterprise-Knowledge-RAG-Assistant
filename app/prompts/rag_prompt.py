from langchain_core.prompts import PromptTemplate

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an enterprise knowledge assistant.

RULES:
- Answer ONLY using the provided context.
- If the answer is not explicitly supported, say:
  "I do not have sufficient information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
)
