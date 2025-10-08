import os
import io
import tempfile
import shutil
import streamlit as st
from dotenv import load_dotenv

# LangChain / Vector DB
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------------- Config ----------------
load_dotenv()
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CHROMA_DIR  = "storage/chroma"
SOURCE_DIR  = "data/sources"
CONF_THRESHOLD = 0.20

st.set_page_config(page_title="IBM AI Research Agent", page_icon="🤖", layout="wide")

# Ensure folders exist
os.makedirs(SOURCE_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# ---------------- Hero ----------------
st.markdown(
    """
    <h1 style="margin-bottom:0">🤖 IBM AI Research Agent</h1>
    <p style="color:#a4a7ae;margin-top:6px">
      Ask questions about your uploaded PDFs (Document Mode) or anything at all (Global AI Mode).
      In Document Mode, answers are grounded in your files with citations.
    </p>
    """,
    unsafe_allow_html=True,
)

# ---------- Page switcher (swipe-like) ----------
if "page" not in st.session_state:
    st.session_state.page = 0  # 0 = Document Mode, 1 = Global Mode

nav_cols = st.columns([1, 8, 1])
with nav_cols[0]:
    if st.button("◀️", use_container_width=True):
        st.session_state.page = (st.session_state.page - 1) % 2
with nav_cols[1]:
    choice = st.radio(
        "Mode",
        options=["📄 Document Mode", "🌍 Global AI Mode"],
        index=st.session_state.page,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state.page = 0 if choice.startswith("📄") else 1
with nav_cols[2]:
    if st.button("▶️", use_container_width=True):
        st.session_state.page = (st.session_state.page + 1) % 2

st.divider()

# ---------------- Helpers ----------------
def chunk_pdfs(file_objs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for f in file_objs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = f.getvalue() if hasattr(f, "getvalue") else f.read()
            tmp.write(content)
            tmp.flush()
            docs.extend(PyPDFLoader(tmp.name).load())
    return splitter.split_documents(docs)

def build_ephemeral_retriever(chunks, k):
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    db = Chroma.from_documents(chunks, embeddings)  # in-memory
    return db.as_retriever(search_kwargs={"k": k}), db

def build_persistent_retriever(k):
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": k}), db

def format_sources(docs):
    lines = []
    for i, d in enumerate(docs, 1):
        src = os.path.basename(d.metadata.get("source", "uploaded.pdf"))
        page = d.metadata.get("page", None)
        lines.append(f"[{i}] {src}" + (f" (page {page})" if page is not None else ""))
    return "\n".join(lines)

def answer_grounded(query, retriever, llm, k):
    try:
        db = retriever.vectorstore
        docs_scores = db.similarity_search_with_score(query, k=k)
        top_docs = [d for d, s in docs_scores]
        top_scores = [s for d, s in docs_scores]
    except Exception:
        top_docs = retriever.get_relevant_documents(query)
        top_scores = []

    if not top_docs:
        return "I don't know based on the provided documents.", [], True

    ctx = "\n\n".join(f"[{i+1}] " + d.page_content for i, d in enumerate(top_docs))
    sys = (
        "You are a careful research assistant. Use ONLY the provided context. "
        "Cite facts with [1], [2], ... matching the chunks. If not in context, say you don't know."
    )
    user = f"Context:\n{ctx}\n\nQuestion: {query}\nRemember to cite like [1],[2]."
    out = llm.invoke([{"role": "system", "content": sys}, {"role": "user", "content": user}]).content

    low_conf = (len(top_scores) > 0 and min(top_scores) > CONF_THRESHOLD)
    return out, top_docs, low_conf

def answer_global(query, llm):
    sys = "You are a helpful, concise assistant."
    return llm.invoke([{"role": "system", "content": sys}, {"role": "user", "content": query}]).content

def status_chip(text, kind="ok"):
    colors = {"ok": "#10b981", "warn": "#f59e0b", "info": "#3b82f6"}
    st.markdown(
        f"""
        <span style="
            display:inline-block;padding:4px 10px;border-radius:999px;
            font-size:12px;background:{colors.get(kind,'#3b82f6')};color:white;margin-bottom:8px;">
            {text}
        </span>
        """,
        unsafe_allow_html=True,
    )

def persist_uploads_to_index(uploads):
    """
    Save uploaded PDFs into data/sources and add them to the persistent Chroma index.
    """
    saved_files = []
    for f in uploads:
        name = f.name if hasattr(f, "name") else "upload.pdf"
        safe_name = name.replace("/", "_").replace("\\", "_")
        dest_path = os.path.join(SOURCE_DIR, safe_name)
        with open(dest_path, "wb") as out:
            out.write(f.getvalue() if hasattr(f, "getvalue") else f.read())
        saved_files.append(dest_path)

    # Build embeddings & persist
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    # Load & chunk newly saved files, then add_documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    new_docs = []
    for path in saved_files:
        new_docs.extend(PyPDFLoader(path).load())
    chunks = splitter.split_documents(new_docs)

    if chunks:
        db.add_documents(chunks)
        try:
            db.persist()  # not strictly needed on newer Chroma, but harmless
        except Exception:
            pass

    return [os.path.basename(p) for p in saved_files], len(chunks)

# ---------------- Page 0: Document Mode ----------------
def page_document_mode():
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("How this works")
        st.markdown(
            """
            **Docs-only (grounded)** — answers **only** from your PDFs, with citations.  
            **Hybrid** — tries docs first; if confidence is low, falls back to general LLM (clearly labeled).
            """
        )
        uploads = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

        # NEW: Persist uploads to index (optional)
        if uploads:
            if st.button("📌 Persist uploads to index", use_container_width=True):
                with st.spinner("Indexing uploaded PDFs..."):
                    saved, count = persist_uploads_to_index(uploads)
                st.success(f"Added {len(saved)} file(s) to the index ({count} chunks).")
                st.write("Files:", ", ".join(saved))

        mode = st.radio("Answer mode", ["Docs-only (grounded)", "Hybrid (fallback to LLM)"])
        k = st.slider("Max sources per answer", 2, 6, 4)
        st.caption("Tips: Drop annual reports, whitepapers, specs. Use the button above to persist uploads.")

        st.markdown("**Example questions**")
        ex_cols = st.columns(3)
        example_labels = ["Summarize the report", "Key metrics", "Growth drivers"]
        example_queries = [
            "Summarize the IBM AI report in 3 bullet points.",
            "What are the key performance metrics in the report?",
            "What are the main growth drivers mentioned?"
        ]
        for col, label, query_text in zip(ex_cols, example_labels, example_queries):
            with col:
                if st.button(label, use_container_width=True):
                    st.session_state["_autofill_query"] = query_text

    with right:
        st.subheader("Ask a question about your PDFs")

        default_q = st.session_state.pop("_autofill_query", "")
        query = st.text_input("Enter your question", value=default_q)

        # NEW: Ready banner when idle / no query yet
        if not query:
            status_chip("✅ Ready and listening…", "ok")

        if query:
            llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
            if uploads:
                chunks = chunk_pdfs(uploads)
                retriever, _ = build_ephemeral_retriever(chunks, k)
            else:
                retriever, _ = build_persistent_retriever(k)

            with st.spinner("Thinking..."):
                grounded_text, docs, low_conf = answer_grounded(query, retriever, llm, k)
                looks_generic = len(query.split()) <= 4
                if mode.startswith("Docs-only") and (looks_generic or low_conf or not docs):
                    st.warning("This looks like a general question. Please use **Hybrid** mode to answer other questions.")
                    status_chip("Grounded mode (no fallback)", "info")
                    st.markdown("### Answer")
                    st.write("I don't know based on the provided documents.")
                elif mode.startswith("Hybrid") and (low_conf or grounded_text.startswith("I don't know")):
                    status_chip("Hybrid fallback (general LLM)", "warn")
                    answer = answer_global(query, llm)
                    st.markdown("### Answer (Hybrid)")
                    st.write(answer)
                else:
                    status_chip("Grounded in your documents", "ok")
                    st.markdown("### Answer")
                    st.write(grounded_text)
                st.expander("Sources").write(format_sources(docs) if docs else "No sources found.")

# ---------------- Page 1: Global AI Mode ----------------
def page_global_mode():
    st.subheader("Ask anything (not limited to your PDFs)")
    st.markdown("Use this for general questions. For document-grounded answers, use **Document Mode**.")
    q2 = st.text_input("Your question (global)")

    if not q2:
        status_chip("✅ Ready and listening…", "ok")

    if q2:
        llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.3)
        status_chip("General answer (no document grounding)", "info")
        with st.spinner("Thinking..."):
            st.markdown("### Answer")
            sys = "You are a helpful, concise assistant."
            out = llm.invoke([{"role":"system","content":sys},{"role":"user","content":q2}]).content
            st.write(out)

# ---------------- Router ----------------
if st.session_state.page == 0:
    page_document_mode()
else:
    page_global_mode()

# ---------------- Footer ----------------
st.markdown(
    """
    <hr style="margin-top:2.5rem;margin-bottom:0.5rem;opacity:0.15;"/>
    <div style="font-size:13px;color:#a4a7ae;text-align:center;margin-bottom:0.8rem;">
      Built by <b>William Wilson</b> ·
      <a href="https://github.com/yourusername" target="_blank" style="color:#7ab8ff;text-decoration:none;">GitHub</a> ·
      Stack: Streamlit · LangChain · ChromaDB · OpenAI
    </div>
    """,
    unsafe_allow_html=True,
)

