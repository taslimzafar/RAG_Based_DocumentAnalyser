from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from rag_core import parse_upload, chunk_text, retrieve, answer_from_chunks

st.set_page_config(page_title="Document Q&A | RAG", page_icon="📚", layout="wide")
st.title("📚 RAG-Based Document Q&A")
st.caption("Ask questions over your documents. Answers are grounded in retrieved text and shown with source chunks.")

with st.sidebar:
    st.header("Add documents")
    uploads = st.file_uploader("Upload PDF, DOCX, TXT, or Markdown", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    st.markdown("---")
    st.caption("Local extractive mode: no API key or cloud LLM required. Review cited text before relying on an answer.")

@st.cache_data(show_spinner=False)
def build_index(items):
    chunks=[]
    for name, data in items:
        text=parse_upload(name,data)
        chunks.extend(chunk_text(text, name))
    return chunks

items=[]
if uploads:
    items=[(f.name, f.getvalue()) for f in uploads]
    try:
        chunks=build_index(items)
        st.success(f"Indexed {len(items)} document(s) into {len(chunks)} searchable chunks.")
    except Exception as exc:
        st.error(f"Could not read a document: {exc}")
        chunks=[]
else:
    chunks=[]
    st.info("Upload one or more documents in the sidebar to start. Try a public-domain policy or technical manual.")

question=st.text_input("Your question", placeholder="What does the document say about ...?")
if st.button("Get answer", type="primary", disabled=not (question.strip() and chunks)):
    hits=retrieve(question, chunks, top_k=4)
    if not hits or hits[0]["score"] <= 0:
        st.warning("I could not find a relevant passage. Try different wording or add a document with the answer.")
    else:
        answer=answer_from_chunks(question, hits)
        st.subheader("Answer (extractive)")
        st.write(answer)
        st.caption("This answer is composed from retrieved sentences. It may be incomplete; check the source excerpts below.")
        st.subheader("Retrieved sources")
        for i, hit in enumerate(hits, 1):
            with st.expander(f"{i}. {hit['source']} · chunk {hit['chunk_id']} · relevance {hit['score']:.3f}", expanded=i==1):
                st.write(hit["text"])

st.markdown("---")
st.header("Supplied dataset overview")
st.caption("The provided CSV has evaluation metadata only; it does not contain document text, queries, or answers. These charts describe those supplied rows, not live retrieval quality.")
dataset_path=Path(__file__).parent/'data'/'dataset.csv'
try:
    df=pd.read_csv(dataset_path)
    c1,c2,c3=st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Documents", f"{df['doc_id'].nunique():,}")
    c3.metric("Mean retrieval score", f"{df['retrieval_score'].mean():.3f}")
    left,right=st.columns(2)
    with left:
        fig,ax=plt.subplots()
        df.groupby('doc_type').size().sort_values().plot(kind='barh',ax=ax,color='#3874cb')
        ax.set(title='Rows by document type',xlabel='Rows',ylabel='Document type')
        st.pyplot(fig,clear_figure=True)
    with right:
        fig,ax=plt.subplots()
        df.groupby('query_type')['answer_quality_score'].mean().sort_values().plot(kind='barh',ax=ax,color='#26a69a')
        ax.set(title='Mean answer quality by query type',xlabel='Mean score',ylabel='Query type')
        st.pyplot(fig,clear_figure=True)
    fig,ax=plt.subplots()
    ax.scatter(df['retrieval_score'],df['answer_quality_score'],alpha=.55,s=18,color='#6c63ff')
    ax.set(title='Retrieval score vs answer quality',xlabel='Retrieval score',ylabel='Answer quality score')
    st.pyplot(fig,clear_figure=True)
    with st.expander("Preview supplied data"):
        st.dataframe(df.head(20),use_container_width=True)
except Exception as exc:
    st.error(f"Could not load data/dataset.csv: {exc}")
