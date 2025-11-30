# src/tools.py

import os
import json
from .rag_pipeline import retrieve_relevant_chunks, build_vector_store

# Load vector store once at startup
_ = build_vector_store()


# ====================================
# 1. RAG Retrieval Tool
# ====================================
def rag_search(query: str) -> str:
    """
    Retrieves relevant chunks using the RAG pipeline.
    Returns text — NOT Python objects.
    """
    try:
        results = retrieve_relevant_chunks(query, top_k=4)
        if not results:
            return "No relevant information found."
        return "\n---\n".join(results)
    except Exception as e:
        return f"Error during RAG search: {e}"


# ====================================
# 2. List Repository Files (Fixed)
# ====================================
def list_repo_files(_: str = "") -> str:
    """
    Lists up to 30 repository files safely.
    Prevents token explosion for Groq.
    """
    try:
        file_list = []
        for root, _, files in os.walk("."):
            for file in files:
                path = os.path.join(root, file)
                file_list.append(path)

        # LIMIT OUTPUT FOR LLM SAFETY
        if len(file_list) > 30:
            file_list = file_list[:30] + ["... (more files truncated)"]

        return "\n".join(file_list)
    except Exception as e:
        return f"Error listing files: {e}"


# ====================================
# 3. Read a File
# ====================================
def read_file(rel_path: str) -> str:
    """
    Reads a file safely with a 2500 character limit.
    """
    try:
        if not os.path.exists(rel_path):
            return f"File not found: {rel_path}"
        with open(rel_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content[:2500]
    except Exception as e:
        return f"Error reading file: {e}"


# ====================================
# 4. Simple Evaluation Tool
# ====================================
def run_eval_on_qa_set(_: str = "") -> str:
    """
    Loads QA evaluation set, tests using agent_answer, and returns scores.
    """
    try:
        path = os.path.join("data", "qa_eval_set.json")
        if not os.path.exists(path):
            return "Evaluation file not found."

        with open(path, "r", encoding="utf-8") as f:
            qa_data = json.load(f)

        # Lazy import to avoid circular dependency
        from .agent_core import agent_answer

        results = []
        for q in qa_data:
            user_q = q["question"]
            ref_keywords = q.get("keywords", [])
            answer = agent_answer(user_q)

            score = sum(1 for k in ref_keywords if k.lower() in answer.lower())
            total = len(ref_keywords)

            results.append({
                "question": user_q,
                "answer": answer,
                "score": f"{score}/{total}"
            })

        return json.dumps({"results": results}, indent=2)

    except Exception as e:
        return f"Error during evaluation: {e}"


# ====================================
# TOOL REGISTRY
# ====================================
TOOL_REGISTRY = {
    "rag_search": rag_search,
    "list_repo_files": list_repo_files,
    "read_file": read_file,
    "run_eval_on_qa_set": run_eval_on_qa_set,
}
