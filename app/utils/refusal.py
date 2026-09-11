LOW_CONFIDENCE_PHRASES = [
    "do not have sufficient information",
    "cannot determine",
    "not provided in the documents",
    "insufficient context"
]

def is_refusal(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in LOW_CONFIDENCE_PHRASES)
