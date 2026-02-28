"""
PDF RAG Chatbot — Streamlit + Claude API + Streaming
=====================================================
Run with: streamlit run app.py

Install:
    pip install streamlit anthropic chromadb pymupdf langchain-text-splitters sentence-transformers
"""

import os
import json
from pathlib import Path

import pdfplumber
import streamlit as st
import anthropic
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 150
CLAUDE_MODEL    = "claude-sonnet-4-5"
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "pdf_docs"

SYSTEM_PROMPT = """You are a precise, helpful assistant that answers questions \
using ONLY the document context provided by the search_documents tool.

Rules:
- Always call search_documents before answering any factual question.
- Cite every claim with [source, p.N] inline.
- If the answer is not in the documents, say "I couldn't find that in the uploaded documents."
- Never hallucinate or use outside knowledge for factual claims.
- Be concise and structured."""

# ─────────────────────────────────────────────
# Cached resources
# ─────────────────────────────────────────────
@st.cache_resource
def get_chroma():
    """ChromaDB with built-in embedding function — no extra install needed."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # Uses chroma's default embedding (downloads ~80MB model automatically)
    return client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


@st.cache_resource
def get_claude(api_key: str):
    """Cached per api_key — fresh client if key changes."""
    return anthropic.Anthropic(api_key=api_key)


def get_api_key() -> str:
    key = st.session_state.get("anthropic_key") or os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        st.error("⚠️ Anthropic API key missing — enter it in the sidebar.")
        st.stop()
    return key


# ─────────────────────────────────────────────
# Ingestion helpers
# ─────────────────────────────────────────────
def extract_pages(pdf_bytes: bytes, filename: str) -> list[dict]:
    import io
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"text": text, "page": i + 1, "source": filename})
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for page in pages:
        for j, split in enumerate(splitter.split_text(page["text"])):
            chunks.append({
                "text":     split,
                "source":   page["source"],
                "page":     page["page"],
                "chunk_id": f"{page['source']}_p{page['page']}_c{j}",
            })
    return chunks


def embed_and_store(chunks: list[dict], progress_bar=None) -> int:
    collection = get_chroma()
    batch_size = 64
    stored     = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]

        # Pass documents only — ChromaDB embeds them automatically
        collection.upsert(
            ids       = [c["chunk_id"] for c in batch],
            documents = texts,
            metadatas = [{"source": c["source"], "page": c["page"]} for c in batch],
        )
        stored += len(batch)
        if progress_bar:
            progress_bar.progress(min(stored / len(chunks), 1.0))

    return stored


def ingest_pdfs(uploaded_files) -> tuple[int, list[str]]:
    all_chunks, filenames = [], []
    for uf in uploaded_files:
        pages  = extract_pages(uf.read(), uf.name)
        chunks = chunk_pages(pages)
        all_chunks.extend(chunks)
        filenames.append(uf.name)

    if not all_chunks:
        return 0, filenames

    pb = st.progress(0.0, text="Embedding chunks locally…")
    n  = embed_and_store(all_chunks, progress_bar=pb)
    pb.empty()
    return n, filenames


# ─────────────────────────────────────────────
# Retrieval tool definition
# ─────────────────────────────────────────────
SEARCH_TOOL = {
    "name": "search_documents",
    "description": "Search the uploaded PDF knowledge base for relevant passages. Always call this before answering.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query":     {"type": "string",  "description": "Natural language search query"},
            "n_results": {"type": "integer", "description": "Passages to retrieve (default 5)", "default": 5},
        },
        "required": ["query"],
    },
}


def run_search(query: str, n_results: int = 5) -> str:
    collection = get_chroma()
    n_results  = st.session_state.get("n_results", n_results)

    # Pass query_texts — ChromaDB embeds automatically
    results = collection.query(
        query_texts = [query],
        n_results   = n_results,
        include     = ["documents", "metadatas", "distances"],
    )
    passages = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        passages.append({
            "text":   doc,
            "source": meta["source"],
            "page":   meta["page"],
            "score":  round(1 - dist, 3),
        })
    return json.dumps(passages, indent=2)


# ─────────────────────────────────────────────
# Streaming chat with tool calling
# ─────────────────────────────────────────────
def stream_chat(messages: list[dict]):
    ac = get_claude(get_api_key())

    while True:
        current_text = ""
        tool_calls   = {}

        with ac.messages.stream(
            model      = CLAUDE_MODEL,
            max_tokens = 2048,
            system     = SYSTEM_PROMPT,
            tools      = [SEARCH_TOOL],
            messages   = messages,
        ) as stream:
            for event in stream:
                etype = event.type

                if etype == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        tool_calls[block.id] = {"name": block.name, "input_str": ""}
                        yield {"type": "tool_start", "name": block.name}

                elif etype == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        current_text += delta.text
                        yield delta.text
                    elif delta.type == "input_json_delta" and tool_calls:
                        last_id = list(tool_calls)[-1]
                        tool_calls[last_id]["input_str"] += delta.partial_json

                elif etype == "content_block_stop":
                    current_text = ""

            final_msg = stream.get_final_message()

        if final_msg.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": final_msg.content})
            tool_results = []
            for block in final_msg.content:
                if block.type == "tool_use":
                    try:
                        inputs = json.loads(tool_calls[block.id]["input_str"]) if tool_calls[block.id]["input_str"] else block.input
                    except Exception:
                        inputs = block.input
                    yield {"type": "tool_result", "query": inputs.get("query", "")}
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     run_search(**inputs),
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            break


# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📚", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@600;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.tool-badge {
    display: inline-block;
    background: rgba(124,106,247,0.15);
    border: 1px solid rgba(124,106,247,0.4);
    color: #a78bfa;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; padding: 2px 10px;
    border-radius: 20px; margin: 4px 0 8px 0;
}
.source-chip {
    display: inline-block; background: #1e293b;
    border: 1px solid #334155; color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; padding: 2px 8px;
    border-radius: 4px; margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    with st.expander("API Keys", expanded=not os.getenv("ANTHROPIC_API_KEY")):
        ak = st.text_input(
            "Anthropic API Key", type="password",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
            help="console.anthropic.com/settings/api-keys",
        )
        if ak:
            st.session_state["anthropic_key"] = ak

    st.caption("✅ Embeddings run **locally via ChromaDB** — no paid API needed")

    st.divider()
    st.markdown("### 📄 Upload PDFs")
    uploaded = st.file_uploader(
        "Drop one or more PDFs", type=["pdf"],
        accept_multiple_files=True, label_visibility="collapsed",
    )

    if uploaded and st.button("🚀 Ingest PDFs", use_container_width=True, type="primary"):
        with st.spinner("Extracting and embedding…"):
            try:
                n, names = ingest_pdfs(uploaded)
                st.session_state["ingested"] = st.session_state.get("ingested", []) + names
                st.success(f"✓ {n} chunks from {len(names)} file(s)")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    if st.session_state.get("ingested"):
        st.markdown("**Indexed files:**")
        for name in set(st.session_state["ingested"]):
            st.markdown(f'<span class="source-chip">📄 {name}</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🛠️ Settings")
    st.session_state["n_results"] = st.slider("Passages retrieved per query", 3, 10, 5)

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    if st.button("💣 Wipe vector store", use_container_width=True):
        try:
            c = chromadb.PersistentClient(path=CHROMA_PATH)
            c.delete_collection(COLLECTION_NAME)
            st.session_state["ingested"] = []
            st.success("Vector store cleared.")
            st.rerun()
        except Exception as e:
            st.error(str(e))

# ── Main chat ─────────────────────────────────
st.markdown("# 📚 PDF RAG Chatbot")
st.markdown("Upload PDFs in the sidebar → Ingest → Ask questions. Claude answers with citations.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your PDFs…"):
    if not (st.session_state.get("anthropic_key") or os.getenv("ANTHROPIC_API_KEY")):
        st.error("Please enter your Anthropic API key in the sidebar.")
        st.stop()
    if not st.session_state.get("ingested"):
        st.warning("Please upload and ingest at least one PDF first.")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state["messages"]
    ]

    with st.chat_message("assistant"):
        placeholder    = st.empty()
        tool_container = st.container()
        full_response  = ""

        for chunk in stream_chat(api_messages):
            if isinstance(chunk, str):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            elif isinstance(chunk, dict):
                if chunk["type"] == "tool_start":
                    with tool_container:
                        st.markdown('<span class="tool-badge">🔍 Searching documents…</span>', unsafe_allow_html=True)
                elif chunk["type"] == "tool_result":
                    with tool_container:
                        st.markdown(f'<span class="tool-badge">📎 Query: {chunk["query"]}</span>', unsafe_allow_html=True)

        placeholder.markdown(full_response)

    st.session_state["messages"].append({"role": "assistant", "content": full_response})