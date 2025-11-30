# src/agent_core.py

import os
import re
import textwrap
from dotenv import load_dotenv
from groq import Groq
from .tools import TOOL_REGISTRY

# =========================================
# 🔑 Load GROQ API key
# =========================================
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =========================================
# GROQ MODEL RESPONSE FUNCTION
# =========================================
def llm_generate(prompt: str, max_new_tokens: int = 200) -> str:
    """
    Uses Groq to generate responses with Llama 3.3 70B model
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # 🟢 Valid Groq model
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error calling Groq API: {e}"


# =========================================
# ENSURE REFLECTION ALWAYS ADDED
# =========================================
def _ensure_reflection(answer: str) -> str:
    if "Reflection:" not in answer:
        answer += "\n\nReflection: I am fairly confident but could improve with deeper reasoning."
    return answer.strip()


# =========================================
# TOOL PARSING (ACTION/INPUT format)
# =========================================
def _parse_action_input_block(text: str):
    pattern = r"ACTION:\s*(\w+)\s*[\r\n]+INPUT:\s*(.*)"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None


def _run_tool(tool_name: str, tool_input: str) -> str:
    tool = TOOL_REGISTRY.get(tool_name)
    if not tool:
        return f"Tool '{tool_name}' not found."
    try:
        return tool(tool_input)
    except Exception as e:
        return f"Error running tool '{tool_name}': {e}"


SYSTEM_DESCRIPTION = textwrap.dedent("""
I am an agentic AI system built using Python, FAISS, and Hugging Face embeddings.
I use a Retrieval-Augmented Generation (RAG) pipeline and Groq Llama 3.3 model for intelligent responses.
I was designed as part of a Ciklum AI Academy–inspired learning experience.
""")


# ===========================================================
#                    MAIN AGENT FUNCTION
# ===========================================================
def agent_answer(user_query: str) -> str:
    q = user_query.lower().strip()

    # -----------------------------------------------------------------
    # 1. TOOL CALL FORMAT (ACTION / INPUT) — For ReAct-style workflows
    # -----------------------------------------------------------------
    direct = _parse_action_input_block(user_query)
    if direct:
        tool_name, tool_input = direct
        obs = _run_tool(tool_name, tool_input)
        prompt = f"""
        Tool output:
        {obs}

        Summarize clearly in 3–5 sentences.
        Do NOT mention tools or internal code.
        End with: Reflection: <confidence>.
        """
        return _ensure_reflection(llm_generate(prompt))


    # -----------------------------------------------------------------
    # 2. LINKEDIN POST — SMART TOPIC DETECTION (NO multiple elifs)
    # -----------------------------------------------------------------
    if "linkedin" in q or "social post" in q or "post" in q:
        prompt = f"""
        The user said:
        "{user_query}"

        Your task:
        ➤ Generate a LinkedIn-style post based on the TOPIC detected above.
        MUST follow this format:

        🔹 Engaging headline with emoji
        🔹 2–3 sentence introduction on the topic
        🔹 Bullet points with key insights or capabilities
        🔹 Since you're an AI agent, mention:
            • RAG using Hugging Face embeddings
            • Groq Llama 3.3 model for generation
            • Project inspired by Ciklum AI Academy
        🔹 Invite people to connect at the end
        🔹 Include 6–8 relevant hashtags
      
        Tone: Professional, human-like, enthusiastic — NOT robotic.
        End with: Reflection: <confidence score>.
        """
        return _ensure_reflection(llm_generate(prompt))


    # -----------------------------------------------------------------
    # 3. TOOL — HOW WERE YOU BUILT? / architecture
    # -----------------------------------------------------------------
    if "how were you built" in q or "architecture" in q:
        rag = _run_tool("rag_search", "Explain how this AI system was built using RAG and tools.")
        prompt = f"""
        Retrieved info:
        {rag}

        Summarize in 3–5 sentences how the AI agent was built:
        • Python + FAISS + Hugging Face embeddings
        • RAG pipeline
        • Tool-calling ability
        • Ciklum AI Academy educational purpose
        End with: Reflection: <confidence>.
        """
        return _ensure_reflection(llm_generate(prompt))


    # -----------------------------------------------------------------
    # 4. TOOL — LIST FILES
    # -----------------------------------------------------------------
    if "list" in q and "file" in q:
        obs = _run_tool("list_repo_files", "")
        prompt = f"""
        These files were found in the repository:
        {obs}

        In 2–4 sentences, describe:
        • What type of project this is
        • What these files indicate
        • Mention 'agentic AI' and 'RAG'
        End with: Reflection: <confidence>.
        """
        return _ensure_reflection(llm_generate(prompt))


    # -----------------------------------------------------------------
    # 5. TOOL — READ FILE
    # -----------------------------------------------------------------
    if q.startswith("read file:"):
        path = user_query.split(":", 1)[1].strip()
        obs = _run_tool("read_file", path)
        prompt = f"""
        Content of {path}:
        {obs}

        Explain this file's purpose in 2–4 sentences.
        End with: Reflection: <confidence>.
        """
        return _ensure_reflection(llm_generate(prompt))


    # -----------------------------------------------------------------
    # 6. TOOL — SELF EVALUATION
    # -----------------------------------------------------------------
    if "evaluation" in q or "self eval" in q or "self-evaluation" in q:
        obs = _run_tool("run_eval_on_qa_set", "")
        prompt = f"""
        Self-evaluation data:
        {obs}

        Summarize:
        • Performance quality
        • What the scores indicate
        • How the system can improve
        End with: Reflection: <confidence>.
        """
        return _ensure_reflection(llm_generate(prompt))


    # -----------------------------------------------------------------
    # 7. DEFAULT — FALLBACK GENERAL ANSWER
    # -----------------------------------------------------------------
    final_prompt = f"""
    {SYSTEM_DESCRIPTION}

    User query:
    {user_query}

    Respond in 2–4 sentences.
    End with: Reflection: <confidence>.
    """
    return _ensure_reflection(llm_generate(final_prompt))
