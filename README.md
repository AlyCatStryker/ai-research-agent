# 🤖 IBM AI Research Agent

**An AI-powered research assistant for PDFs.**  
Upload corporate reports or whitepapers and ask questions in plain English. Choose:

- **Document Mode (grounded)** – answers *only* from your PDFs, with citations.
- **Hybrid** – tries your PDFs first; if confidence is low, falls back to general LLM.
- **Global AI Mode** – ask anything (not limited to your PDFs).

https://github.com/AlyCatStryker/ibm-ai-research-agent

---

## ✨ Features

- 📄 Multi-PDF upload (in-memory) + **Persist to Index** button  
- 🔎 Retrieval-Augmented Generation (RAG) via **ChromaDB** + OpenAI embeddings  
- 🧠 Hybrid fallback to general LLM when docs don’t have the answer  
- 📚 Citations (file + page) so answers are traceable  
- 🧭 Clean UI with **Document Mode** vs **Global AI Mode** page navigation  
- ✅ **Ready and listening…** status when idle  
- 🧰 Built with **Streamlit, LangChain, OpenAI, ChromaDB**

---

## 🧱 Tech Stack

- Frontend: **Streamlit**
- AI Orchestration: **LangChain**
- LLM: **OpenAI GPT (gpt-4o-mini configurable via `.env`)**
- Embeddings: **text-embedding-3-small**
- Vector DB: **ChromaDB**
- Language: **Python 3.10+**

---

## 🚀 Quickstart

1) **Clone & create venv**
```bash
git clone https://github.com/yourusername/ibm-ai-research-agent.git
cd ibm-ai-research-agent
python3 -m venv .venv
source .venv/bin/activate

