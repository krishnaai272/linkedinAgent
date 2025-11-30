# src/cli_demo.py

from src.agent_core import agent_answer

def main():
    print(
        """
============================================================
Agentic RAG Repo Inspector – CLI Demo
============================================================

This is a small agentic AI system that:
• Uses Retrieval-Augmented Generation (RAG) over a local knowledge base.
• Can inspect its own repository (list and read files).
• Can run a simple self evaluation.
• Always finishes answers with a reflection.

It is built with Python and local Hugging Face models — created
as part of a learning experience inspired by the Ciklum AI Academy.

Type a question and press Enter.
Examples:
- What is RAG and why is it used here?
- How were you built?
- Please list repository files.
- Read file: README.md
- Run a self evaluation.
Type 'exit' or 'quit' to leave.
============================================================
"""
    )

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Goodbye!")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Exiting. Goodbye!")
            break

        # Smart parsing for common intents:
        if user_input.lower().startswith("read file:"):
            path = user_input.split(":", 1)[1].strip()
            agent_query = f"Use the read_file tool to read the file '{path}' and summarize its purpose."
        elif "list" in user_input.lower() and "file" in user_input.lower():
            agent_query = "Use the list_repo_files tool to show important files in this repository."
        elif "evaluation" in user_input.lower() or "self evaluation" in user_input.lower():
            agent_query = (
                "ACTION: run_eval_on_qa_set\nINPUT: "
			)
        else:
            agent_query = user_input

        print("\nAgent is thinking...\n")
        answer = agent_answer(agent_query)
        print(f"Agent>\n{answer}\n")

if __name__ == "__main__":
    main()
