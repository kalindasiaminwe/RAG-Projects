
import os
import streamlit as st
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from dotenv import load_dotenv

# ── Load environment variables ONCE ──────────────────────────────────────────
load_dotenv()

st.set_page_config(page_title="Zambezi Voice RAG", page_icon="🎙️")
st.title("🎙️ Zambezi Voice — Hybrid Search")
st.caption("Pinecone · BM25 · mxbai-embed-large · Llama 3.1")

@st.cache_resource(show_spinner="Loading models and index…")
def build_chain():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    bm25 = BM25Encoder().default()

    # ── Pinecone setup ───────────────────────────────────────────────────────
    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise ValueError("❌ PINECONE_API_KEY not found. Check your .env file.")

    pc = Pinecone(api_key=api_key)
    hybrid_index = pc.Index("lawpal-hybrid")

    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25,
        index=hybrid_index,
        top_k=4,
        alpha=0.5,
        text_key="text",
    )

    llm = ChatOllama(model="llama3.1")

    prompt = PromptTemplate.from_template(
        """Use the following context to answer the question.
If you don't know the answer, say that you don't know.

Context:
{context}

Question:
{question}
"""
    )

    def format_docs(docs):
        return "\\n\\n".join(doc.page_content for doc in docs)

    chain = (
        RunnableParallel(
            context=retriever | RunnableLambda(format_docs),
            question=RunnablePassthrough(),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


chain, retriever = build_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask about the Zambezi Voice research…"):
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.expander("📄 Retrieved chunks", expanded=False):
            docs = retriever.invoke(question)
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "unknown")
                st.markdown(f"**Chunk {i}** — `{source}`")
                st.caption(
                    doc.page_content[:400]
                    + ("…" if len(doc.page_content) > 400 else "")
                )

        placeholder = st.empty()
        full_response = ""

        for chunk in chain.stream(question):
            full_response += chunk
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

with st.sidebar:
    st.header("⚙️ Settings")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**Index:** `lawpal-hybrid`")
    st.markdown("**Embeddings:** `mxbai-embed-large`")
    st.markdown("**LLM:** `llama3.1`")
    st.markdown("**top_k:** 4 · **alpha:** 0.5")
    st.markdown("**Metric:** dotproduct")
