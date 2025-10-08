import os
import io
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

# LangChain / OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# Text splitter (v0.2 moved to a separate package; we support both)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# -----------------------------
# Setup & constants
# -----------------------------
load_dotenv()

# Prefer Streamlit Secrets if present
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

if not OPENAI_API_KEY:
    st.warning(
        "Missing OPENAI_API_KEY. Add it in Streamlit secrets (Deploy → Settings → Secrets) "
        "or set an environment variable."
    )

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "data", "sources")
FAISS_DIR = os.path.join(BASE_DIR, "storage", "faiss")

os.makedirs(SOURCE_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def make_llm() -> ChatOpenAI:
    """Create a deterministic chat model."""
    return ChatOpenAI(model=MODEL, temperature=0, api_key=OPENAI_API_KEY)

def _split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def _save_upload_to_disk(upload) -> str:
    """Save an uploaded file-like object into SOURCE_DIR and return its path."""
    name = getattr(upload, "name", "upload.pdf")
    safe = name.replace("/", "_").replace("\\", "_")
    dest = os.path.join(SOURCE_DIR, safe)
    # BytesIO (Streamlit) supports getbuffer()/getvalue()
    data = upload.getvalue() if hasattr(upload, "getvalue") else upload.read()
    with open(dest, "wb") as f:
        f.write(data)
    return dest

def _load_pdfs(paths: List[str]):
    docs = []
    for p in paths:
        try:
            docs.extend(PyPDFLoader(p).load())
        except Exception as e:
            st.error(f"Failed to read {os.path.basename(p)}: {e}")
    return docs

def build_ephemeral_retriever_from_uploads(uploads, k: int):
    """Build an in-memory FAISS index from *current* uploads (not persisted)."""
    if not uploads:
        return None, None, []

    saved_paths = [_save_upload_to_disk(u) for u in uploads]
    docs = _load_pdfs(saved_paths)
    if not docs:
        return None, None, saved_paths
    chunks = _split_docs(docs)

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": k})

    return retriever, db, saved_paths

def build_persistent_retriever(k: int):
    """Load or bootstrap a disk-persisted FAISS index."""
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
    try:
        db = FAISS.load_local(
            FAISS_DIR, embeddings, allow_dangerous_deserialization=True
        )
    except Exception:
        # create a tiny placeholder index so .as_retriever works
        db = FAISS.from_texts([" "], embeddings)
        db.save_local(FAISS_DIR)
    retriever = db.as_retriever(search_kwargs={"k": k})
    return retriever, db

def persist_uploads_to_index(uploads) -> Tuple[List[str], int]:
    """
    Save uploads to disk and add chunks into the *persistent* FAISS index.
    Returns (saved_filenames, num_chunks_added).
    """
    if not uploads:
        return [], 0

    saved = [_save_upload_to_disk(u) for u in uploads]
    docs = _load_pdfs(saved)
    if not docs:
        return [os.path.basename(p) for p in saved], 0

    chunks = _split_docs(docs)
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)

    # Load or create FAISS, then add docs and save back
    try:
        db = FAISS.load_local(
            FAISS_DIR, embeddings, allow_dangerous_deserialization=True
        )
    except Exception:
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(FAISS_DIR)
        return [os.path.basename(p) for p in saved], len(chunks)

    db.add_documents(chunks)
    db.save_local(FAISS_DIR)
    return [os.path.basename(p) for p in saved], len(chunks)

def format_sources(docs: List):
    items = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        name = os.path.basename(meta.get("source", ""))
        page = meta.get("page")
        label = f"[{i}] {name}" if name else f"[{i}]"
        if page is not None:
            label += f" · page {page+1}"
        items.append(label)
    return items

# -----------------------------
# QA logic
# -----------------------------
def answer_from_docs(query: str, retriever, llm: ChatOpenAI):
    """Strict, grounded answer using only retrieved context."""
    # fetch docs
    top_docs = retriever.get_relevant_documents(query) if retriever else []
    if not top_docs:
        return (
            "I don't know based on the provided documents.",
            [],
            True,  # treat as low confidence
        )

    ctx = "\n\n".join(f"[{i+1}] " + d.page_content for i, d in enumerate(top_docs))
    sys = (
        "You are a careful research assistant. Use ONLY the provided context to answer. "
        "Cite facts using [1], [2], ... corresponding to the provided chunks. "
        "If the answer is not in the context, say you don't know."
    )
    user = f"Context:\n{ctx}\n\nQuestion: {query}\nRemember to cite like [1], [2]."

    out = llm.invoke(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    ).content

    # very short / generic queries are likely off-topic in docs-only mode
    looks_generic = len(query.strip().split()) <= 4
    return out, top_docs, looks_generic

def answer_hybrid(query: str, retriever, llm: ChatOpenAI):
    """
    Try grounded answer first. If not confident, fall back to general model.
    """
    grounded, docs, low_conf = answer_from_docs(query, retriever, llm)
    if ("I don't know" in grounded) or low_conf:
        # Fallback – general model (clearly labeled)
        sys = (
            "You are a helpful general AI assistant. If the user question mentions any "
            "documents or sources, answer broadly without citing local documents."
        )
        general = llm.invoke(
            [{"role": "system", "content": sys}, {"role": "user", "content": query}]
        ).content
        return general, docs, True  # True indicates this came from fallback
    return grounded, docs, False

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="IBM AI Research Agent", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
      .smallmuted {color:#9aa0a6; font-size:0.9rem;}
      .chip {display:inline-block; padding:4px 10px; border-radius:999px; font-size:0.8rem; margin-left:6px;}
      .chip-green {background:#163b26; color:#b7ffd1; border:1px solid #2b8a3e;}
      .chip-blue {background:#0d2a3a; color:#bde4ff; border:1px solid #1d6ea3;}
      footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 IBM AI Research Agent")
st.caption(
    "Ask questions about your uploaded PDFs (**Document Mode**) or anything at all (**Global AI Mode**). "
    "In Document Mode, answers are grounded in your files with citations."
)

tabs = st.tabs(["📄 Document Mode", "🌍 Global AI Mode"])

# Shared controls
if "k" not in st.session_state:
    st.session_state.k = 4

# -----------------------------
# DOCUMENT MODE TAB
# -----------------------------
with tabs[0]:
    left, right = st.columns([0.9, 1.1], gap="large")

    with left:
        st.subheader("How this works")
        st.markdown(
            """
            **Docs-only (grounded)** — answers only from your PDFs, with citations.  
            **Hybrid (fallback to LLM)** — tries docs first; if confidence is low, falls back to general AI (clearly labeled).
            """,
        )

        uploads = st.file_uploader(
            "Upload PDF(s)",
            accept_multiple_files=True,
            type=["pdf"],
            help="Drop annual reports, whitepapers, specs. Add them to a persistent index with the button below.",
        )

        col_persist1, col_persist2 = st.columns([1, 1])
        with col_persist1:
            if st.button("📌 Persist uploads to index"):
                saved, n_chunks = persist_uploads_to_index(uploads or [])
                if saved:
                    st.success(
                        f"Indexed {n_chunks} chunks from: "
                        + ", ".join(os.path.basename(s) for s in saved)
                    )
                else:
                    st.info("No files to index, or nothing new to add.")

        mode = st.radio(
            "Answer mode",
            options=["Docs-only (grounded)", "Hybrid (fallback to LLM)"],
            index=0,
            horizontal=False,
        )

        st.slider(
            "Max sources per answer",
            min_value=2,
            max_value=6,
            value=st.session_state.k,
            key="k",
            help="The retriever will fetch this many chunks to ground each answer.",
        )

        st.caption(
            "Tips: Drop annual reports, whitepapers, specs. Re-run ingestion to persist files long-term."
        )

        st.markdown("**Example questions**")
        ex1, ex2, ex3 = st.columns(3)
        if ex1.button("Summarize the report"):
            st.session_state.example_query = "Summarize the report in 3 bullet points."
        if ex2.button("Key metrics"):
            st.session_state.example_query = "What are the key metrics and KPIs mentioned?"
        if ex3.button("Growth drivers"):
            st.session_state.example_query = "What are the top growth drivers and risks?"

    with right:
        st.subheader("Ask a question about your PDFs")
        query = st.text_input(
            "Enter your question",
            value=st.session_state.get("example_query", ""),
            label_visibility="collapsed",
        )

        # Build ephemeral retriever from current uploads (not persisted), OR persistent index
        llm = make_llm()
        ephemeral_retriever, _, _ = build_ephemeral_retriever_from_uploads(
            uploads or [], st.session_state.k
        )
        if ephemeral_retriever is None:
            persistent_retriever, _db = build_persistent_retriever(st.session_state.k)
            retriever = persistent_retriever
        else:
            retriever = ephemeral_retriever

        if not query.strip():
            st.info("✅ Ready and listening…")
        else:
            # Badge showing which engine is used
            if mode.startswith("Docs-only"):
                st.markdown('<span class="chip chip-green">Grounded in your documents</span>', unsafe_allow_html=True)
                answer, docs, generic = answer_from_docs(query, retriever, llm)
                st.markdown("### Answer")
                st.write(answer)

                if generic and "I don't know" in answer:
                    st.info("This looks like a general question. Please switch to **Hybrid** mode to answer non-document questions.")

                with st.expander("Sources"):
                    items = format_sources(docs)
                    if items:
                        st.markdown("\n".join(f"- {x}" for x in items))
                    else:
                        st.write("No sources.")
            else:
                st.markdown('<span class="chip chip-blue">Hybrid (may fall back to general AI)</span>', unsafe_allow_html=True)
                answer, docs, used_fallback = answer_hybrid(query, retriever, llm)
                st.markdown("### Answer (Hybrid)")
                st.write(answer)
                with st.expander("Sources"):
                    items = format_sources(docs)
                    if items:
                        st.markdown("\n".join(f"- {x}" for x in items))
                    else:
                        st.write("No document sources used.")
                if used_fallback:
                    st.caption('Answered using general AI due to low confidence in docs.')

# -----------------------------
# GLOBAL AI MODE TAB
# -----------------------------
with tabs[1]:
    st.subheader("Ask anything (not limited to your documents)")
    q = st.text_input("Enter your question for the global model", key="global_q")
    if not q.strip():
        st.info("✅ Ready and listening…")
    else:
        llm = make_llm()
        st.markdown('<span class="chip chip-blue">General answer</span>', unsafe_allow_html=True)
        sys = "You are a helpful, concise assistant."
        out = llm.invoke(
            [{"role": "system", "content": sys}, {"role": "user", "content": q}]
        ).content
        st.markdown("### Answer")
        st.write(out)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    '<span class="smallmuted">Built by <b>William Wilson</b> · Stack: Streamlit · LangChain · FAISS · OpenAI</span>',
    unsafe_allow_html=True,
)

