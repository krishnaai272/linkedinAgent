# 🧠 Agentic AI System with RAG + Groq + Tools  
A complete AI-Agentic system built using **Python**, **Hugging Face embeddings**, and **Groq’s LLM API** — designed as part of a **learning initiative inspired by the Ciklum AI Academy**.

---

## 🚀 Project Features  
This project demonstrates **real-world Agentic AI capabilities**, including:

✔ Retrieval-Augmented Generation (RAG) using a local knowledge base  
✔ Groq Llama 3.3 model for intelligent generation  
✔ Repository inspection tools:
   - `list_repo_files` – view project files  
   - `read_file` – inspect any file in real-time  
✔ Self-evaluation using `qa_eval_set.json`  
✔ Automatic **reflection after every answer**  
✔ CLI-based interactive agent  

---

## 📁 Project Structure  

agentProject/
│
├── data/
│ ├── knowledge_base.md
│ ├── qa_eval_set.json
│
├── src/
│ ├── agent_core.py
│ ├── rag_pipeline.py
│ ├── tools.py
│ ├── cli_demo.py
│ ├── evaluation.py
│ ├── post_generator.py
│
├── tests/
│ ├── test_rag_pipeline.py
│ ├── test_agent_basic.py
│
├── architecture.mmd
├── requirements.txt
└── README.md

yaml


---

## 🧪 How to Run the Agent  

```bash
pip install -r requirements.txt
python -m src.cli_demo
🧠 Example Prompts to Try
swift
Copy code
How were you built?
Please list repository files.
Read file: README.md
What is RAG and why is it used here?
Generate a LinkedIn post about yourself.
Run a self evaluation.
🔧 Technologies Used
Component	Technology
Embeddings	Hugging Face – MiniLM
Generation	Groq API – Llama 3.3 model
Interface	Python CLI
Evaluation	Keyword-based scoring
Agent Pattern	ReAct-style reasoning & reflection

🧾 Evaluation
Run a simple QA-based evaluation:


python -m src.evaluation
📢 LinkedIn Post Generation (Agent Output)
The agent can generate a professional LinkedIn post automatically.
Try this inside the CLI:


Generate a LinkedIn post about your purpose and tools used.
🎯 Purpose of This Project
This project was designed as a capstone-style learning exercise inspired by the Ciklum AI Academy, to demonstrate modern AI techniques:

Agentic AI workflows

Tool-calling mechanisms

Retrieval-Augmented Generation (RAG)

Self-reflection & evaluation

Practical real-world AI behavior

🤝 Connect & Explore
Let’s connect if you're exploring:

Agentic AI 🚀

RAG Systems 📚

Groq API 🤖

Hugging Face Models 🧠

Python AI Projects 🧩

Built with passion for AI engineering & learning!


---







