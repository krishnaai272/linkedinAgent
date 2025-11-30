<!-- data/knowledge_base.md -->
# Knowledge Base: Agentic RAG System for Ciklum AI Academy

## Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a technique where a language model is combined with a retrieval system.
Instead of relying only on its internal parameters, the model first retrieves relevant documents or chunks from a
knowledge base and then generates an answer conditioned on that external context.

RAG is useful when:
- You want more **grounded** and **up-to-date** answers.
- You need to control or inspect the knowledge source.
- You want to avoid hallucinations by pointing the model at trusted documents.

In this project, RAG is implemented using:
- A local markdown file (`data/knowledge_base.md`) as the knowledge source.
- A sentence-transformer embedding model.
- A simple in-memory vector store with cosine similarity.

---

## Agentic AI: Tools, Reasoning, Reflection

An **agentic AI system** goes beyond a single LLM call.
It can:
1. **Reason** about the user’s goal.
2. Decide whether it needs to use external **tools**.
3. **Call tools** to retrieve data, inspect files, or run evaluations.
4. **Reflect** on its own answer and confidence.

Typical agent tools include:
- Retrieval tools (RAG).
- File system tools (list files, read files).
- Evaluation or analytics tools.
- Custom domain-specific tools (APIs, simulations, etc.).

In this project, the agent:
- Uses tools like `rag_search`, `list_repo_files`, `read_file`, and `run_eval_on_qa_set`.
- Follows a ReAct-style loop: Reason → Act (tool) → Observe → Answer.
- Appends a short **Reflection** sentence to each reply.

---

## Ciklum AI Academy

The **Ciklum AI Academy** is a learning and upskilling initiative that helps engineers gain practical experience
with modern AI techniques. Participants build hands-on projects involving LLMs, RAG, and agentic workflows.

This project is designed as if it were part of the Ciklum AI Academy curriculum:
- You practice building an end-to-end agent system.
- You integrate RAG, tools, and reflection into a single pipeline.
- You generate a professional social post describing your work.

The agent is aware that it was built in this educational context and can talk about:
- The AI Academy.
- Why RAG and agentic patterns are important for real-world AI systems.
- How the project is structured and which technologies it uses.

---

## Large Language Models (LLMs)

A **Large Language Model (LLM)** is a neural network trained on a large text corpus to predict the next tokens.
LLMs are capable of:
- Answering questions.
- Summarizing documents.
- Writing code.
- Acting as the “reasoning core” of an agent.

In this project:
- The LLM is `google/flan-t5-base`, a relatively small, instruction-tuned model.
- It is used locally via the `transformers` text2text-generation pipeline.
- The agent prompts the model with system instructions and context from retrieved documents.

---

## Embeddings and Vector Stores

**Embeddings** map text into vectors in a high-dimensional space.
Text with similar meaning tends to have embeddings that are close to each other, enabling semantic search.

Key points:
- We use `sentence-transformers/all-MiniLM-L6-v2` to compute embeddings.
- The knowledge base is split into chunks (paragraphs).
- Each chunk is embedded and stored in an in-memory vector store (numpy array).
- For a user query, we:
  - Compute the query embedding.
  - Compute cosine similarity with each chunk.
  - Select the top-k most relevant chunks as RAG context.

This simple vector store is sufficient for small projects and demos.

---

## Evaluation: Accuracy, Relevance, Clarity

Evaluating an agent is important to understand its performance.
Here, we use a small, heuristic-based evaluation:

- A JSON file `data/qa_eval_set.json` contains:
  - `question`: an evaluation question.
  - `reference_answer`: a reference description.
  - `keywords`: a list of keywords that should appear in a good answer.

The evaluation harness:
- Calls the agent for each question.
- Checks whether all keywords appear in the answer.
- Computes simple scores such as:
  - How many questions had all keywords present.
  - A rough "keyword coverage" ratio.

While simplistic, this approach illustrates how an agent can **evaluate itself** in a basic but automated way.

---

## This Repository as an Inspectable Context

The agent can inspect its own repository via tools:
- `list_repo_files` lists important files such as:
  - `README.md`
  - `architecture.mmd`
  - `src/agent_core.py`
- `read_file` allows the agent to read file contents up to a safe size limit.

With these tools, the agent can:
- Describe how it was built.
- Explain what each component does.
- Generate a social-media style post summarizing the project for platforms like LinkedIn.

This self-inspection is part of what makes the system **agentic**: it uses tools to better understand itself and its environment.
