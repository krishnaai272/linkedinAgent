# tests/test_agent_basic.py
from src.agent_core import agent_answer


def test_agent_answer_non_empty():
    question = "What is this project about?"
    answer = agent_answer(question)
    assert isinstance(answer, str), "Agent answer should be a string."
    assert len(answer.strip()) > 0, "Agent answer should not be empty."

    lower_answer = answer.lower()
    keywords = ["agent", "rag", "ai academy", "ciklum"]
    assert any(k in lower_answer for k in keywords), (
        "Answer should mention at least one of: 'agent', 'RAG', 'AI Academy', 'Ciklum'."
    )
