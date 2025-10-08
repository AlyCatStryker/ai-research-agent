# 🤖 IBM AI Research Agent

A **Streamlit + LangChain-powered AI research assistant** that answers questions from uploaded PDFs using OpenAI's GPT models.  
Users can switch between two powerful modes — **Document Mode** (answers only from uploaded PDFs) and **Global AI Mode** (fallback to general AI).

---

## 🚀 Features
- 📂 Upload one or multiple PDFs (reports, research papers, etc.)
- 🧠 Ask natural questions about document contents
- ⚡ Choose between:
  - **Document Mode** – answers only from your PDFs with citations
  - **Hybrid (Global AI Mode)** – blends document knowledge with general AI reasoning
- 🗂️ Local document indexing via **ChromaDB**
- 🧩 Built with **Streamlit**, **LangChain**, **OpenAI API**, and **ChromaDB**

---

## 🛠️ Tech Stack
- Python 3.10  
- Streamlit  
- LangChain  
- OpenAI API  
- ChromaDB  
- dotenv  

---

## 🧑‍💻 Run Locally

```bash
# Clone the repo
git clone https://github.com/AlyCatStryker/ai-research-agent.git
cd ai-research-agent

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

