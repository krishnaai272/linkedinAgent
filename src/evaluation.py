# src/evaluation.py
import json
import os
from typing import Dict, List, Optional

from .agent_core import agent_answer


def _get_repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _get_qa_path() -> str:
    return os.path.join(_get_repo_root(), "data", "qa_eval_set.json")


def _load_qa_eval_set() -> List[Dict]:
    path = _get_qa_path()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_qa_set(max_questions: Optional[int] = None) -> Dict:
    """
    Evaluate the agent on the QA eval set.

    Args:
        max_questions: Optional maximum number of questions to evaluate.

    Returns:
        A report dictionary with per-question and overall stats.
    """
    qa_items = _load_qa_eval_set()
    if max_questions is not None:
        qa_items = qa_items[:max_questions]

    results: List[Dict] = []
    num_with_all_keywords = 0

    for idx, item in enumerate(qa_items, start=1):
        question = item.get("question", "")
        reference = item.get("reference_answer", "")
        keywords = item.get("keywords", [])

        print(f"Evaluating Q{idx}: {question}")
        answer = agent_answer(question)
        answer_lower = answer.lower()

        keyword_hits = [kw for kw in keywords if kw.lower() in answer_lower]
        all_keywords_present = len(keyword_hits) == len(keywords)
        if all_keywords_present:
            num_with_all_keywords += 1

        results.append(
            {
                "question": question,
                "reference_answer": reference,
                "agent_answer": answer,
                "keywords": keywords,
                "keyword_hits": keyword_hits,
                "all_keywords_present": all_keywords_present,
            }
        )

    num_questions = len(qa_items)
    keyword_coverage = (num_with_all_keywords / num_questions) if num_questions > 0 else 0.0

    report: Dict = {
        "overall": {
            "num_questions": num_questions,
            "num_with_all_keywords": num_with_all_keywords,
            "keyword_coverage": keyword_coverage,
        },
        "results": results,
    }
    return report


def main():
    """
    Run the evaluation and print a human-readable report.
    """
    report = evaluate_qa_set()
    overall = report["overall"]
    results = report["results"]

    print("\n=== Evaluation Summary ===")
    print(f"Number of questions: {overall['num_questions']}")
    print(f"Questions with all keywords present: {overall['num_with_all_keywords']}")
    print(f"Keyword coverage: {overall['keyword_coverage']:.2f}")

    print("\n=== Per-question details ===")
    for idx, r in enumerate(results, start=1):
        print(f"\nQ{idx}: {r['question']}")
        print(f"Reference: {r['reference_answer']}")
        short_answer = r["agent_answer"][:400].replace("\n", " ")
        if len(r["agent_answer"]) > 400:
            short_answer += " ..."
        print(f"Agent answer (trimmed): {short_answer}")
        print(f"Keywords: {', '.join(r['keywords'])}")
        print(f"Keyword hits: {', '.join(r['keyword_hits'])}")
        print(f"All keywords present: {r['all_keywords_present']}")


if __name__ == "__main__":
    main()
