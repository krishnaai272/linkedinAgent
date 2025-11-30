# src/post_generator.py
from .agent_core import agent_answer


def generate_social_post() -> str:
    """
    Ask the agent to generate a LinkedIn-style post about itself.
    """
    prompt = (
        "You are an AI agent that uses RAG, tools, and reflection. "
        "Write a professional LinkedIn-style post (5–7 sentences) where YOU, the agent, "
        "introduce yourself in the first person. Explain at a high level:\n"
        "- What you do (RAG over a local knowledge base, inspecting the repo, evaluating yourself).\n"
        "- How you were built (Python, local Hugging Face models like flan-t5 and MiniLM, simple vector store, tools, agentic loop).\n"
        "- That you were created as part of a learning project in the Ciklum AI Academy (or AI Academy at Ciklum).\n"
        "Keep the tone authentic and concise, suitable for sharing on LinkedIn. "
        "Focus on clarity, not buzzwords. "
        "Return only the post text, no extra explanations."
    )
    response = agent_answer(prompt)
    # The agent_answer will include a Reflection line; for a clean social post,
    # we can strip the reflection when returning from this helper.
    lines = response.splitlines()
    filtered_lines = [line for line in lines if not line.strip().lower().startswith("reflection:")]
    post_text = "\n".join(filtered_lines).strip()
    return post_text


def main():
    post = generate_social_post()
    print(
        "====================================\n"
        "Generated LinkedIn-style Post\n"
        "====================================\n"
    )
    print(post)
    print("\n====================================")


if __name__ == "__main__":
    main()
