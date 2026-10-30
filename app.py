import io
from pathlib import Path
from collections import Counter
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from rag_core import parse_upload, chunk_text, retrieve, answer_from_chunks

# Page configuration
st.set_page_config(
    page_title="DocuRAG | Intelligent Document Q&A",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
CUSTOM_CSS = """
<style>
    /* Global styles */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    
    /* Modern card containers */
    .stat-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
    }
    
    .answer-box {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.08) 0%, rgba(59, 130, 246, 0.02) 100%);
        border-left: 5px solid #2563EB;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0 25px 0;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-high {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-mid {
        background-color: rgba(245, 158, 11, 0.2);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .badge-low {
        background-color: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-info {
        background-color: rgba(37, 99, 235, 0.2);
        color: #60A5FA;
        border: 1px solid rgba(37, 99, 235, 0.4);
    }
    
    /* Document stat pill */
    .doc-tag {
        background: rgba(100, 116, 139, 0.15);
        border: 1px solid rgba(100, 116, 139, 0.3);
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 0.82rem;
        display: inline-block;
        margin: 3px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "clear_input_flag" in st.session_state and st.session_state.clear_input_flag:
    st.session_state.query_input = ""
    st.session_state.clear_input_flag = False

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚡ **DocuRAG System**")
    st.caption("🔒 100% Local & Private Extractive Intelligence")
    st.markdown("---")
    
    st.subheader("📁 Upload Documents")
    uploads = st.file_uploader(
        "PDF, DOCX, TXT, or Markdown",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Upload files to build your local retrieval corpus. Files are processed entirely in memory."
    )
    
    with st.expander("⚙️ RAG Engine Hyperparameters", expanded=False):
        top_k = st.slider("Top Sources (Top-K)", min_value=1, max_value=8, value=4, step=1,
                          help="Number of most relevant passages retrieved per question.")
        chunk_size = st.slider("Chunk Size (words)", min_value=80, max_value=300, value=180, step=20,
                               help="Word limit for each chunk snippet.")
        chunk_overlap = st.slider("Chunk Overlap (words)", min_value=10, max_value=60, value=35, step=5,
                                  help="Overlap between adjacent chunks to maintain context continuity.")
        max_sentences = st.slider("Max Answer Sentences", min_value=2, max_value=10, value=5, step=1,
                                  help="Maximum number of sentences synthesized in the final extractive summary.")
    
    st.markdown("---")
    st.markdown("💡 **Supported Formats:**")
    st.caption("• Adobe PDF (`.pdf`)\n• Microsoft Word (`.docx`)\n• Plain Text (`.txt`)\n• Markdown (`.md`)")

# Indexing cached function
@st.cache_data(show_spinner=False)
def index_documents(items, c_size, c_overlap):
    chunks = []
    file_stats = {}
    for name, data in items:
        try:
            text = parse_upload(name, data)
            doc_chunks = chunk_text(text, name, size=c_size, overlap=c_overlap)
            chunks.extend(doc_chunks)
            file_stats[name] = {
                "word_count": len(text.split()),
                "chunk_count": len(doc_chunks),
                "char_count": len(text)
            }
        except Exception as err:
            file_stats[name] = {"error": str(err)}
    return chunks, file_stats

items = [(f.name, f.getvalue()) for f in uploads] if uploads else []
chunks = []
file_stats = {}

if items:
    with st.spinner("Indexing and chunking documents..."):
        chunks, file_stats = index_documents(items, chunk_size, chunk_overlap)

# Header Section
col_head_left, col_head_right = st.columns([3, 1])
with col_head_left:
    st.title("📚 Intelligent Document Q&A & Analysis")
    st.markdown(
        "Ask questions grounded strictly in your documents. Answers cite exact source snippets with relevance rankings."
    )
with col_head_right:
    if chunks:
        st.markdown(
            f"""
            <div style="text-align: right; padding-top: 15px;">
                <span class="badge-pill badge-high">● {len(items)} Document(s)</span>
                <span class="badge-pill badge-info">{len(chunks)} Chunks</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="text-align: right; padding-top: 15px;">
                <span class="badge-pill badge-mid">○ Awaiting Documents</span>
            </div>
            """,
            unsafe_allow_html=True
        )

# Main Navigation Tabs
tab_qa, tab_inspector, tab_dataset, tab_about = st.tabs([
    "💬 Document Q&A",
    "🔍 Chunks & Corpus Inspector",
    "📊 Dataset Evaluation & Insights",
    "ℹ️ Architecture & Pipeline"
])

# -------------------------------------------------------------
# TAB 1: DOCUMENT Q&A
# -------------------------------------------------------------
with tab_qa:
    if not chunks:
        st.info("👈 **Get started by uploading one or more files in the sidebar!**\n\nTry uploading technical papers, reports, policies, or contracts.")
    else:
        st.markdown("#### 🔎 Query Your Knowledge Base")
        
        # Suggested prompt pills if available
        st.caption("Quick actions:")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        sample_query = None
        if col_s1.button("📌 Summarize main purpose", use_container_width=True):
            sample_query = "What is the primary purpose and scope of this document?"
        if col_s2.button("🔑 Key findings & conclusions", use_container_width=True):
            sample_query = "What are the key findings, results, or conclusions?"
        if col_s3.button("📋 Requirements & conditions", use_container_width=True):
            sample_query = "What requirements, conditions, or constraints are specified?"
        if col_s4.button("⚡ Action items or next steps", use_container_width=True):
            sample_query = "What are the recommended action items, next steps, or policies?"

        if sample_query:
            st.session_state.query_input = sample_query

        # Input box
        query = st.text_input(
            "Enter your question",
            value=st.session_state.query_input,
            placeholder="e.g. What are the key policies or findings discussed in section 2?",
            label_visibility="collapsed"
        )
        
        col_btn1, col_btn2, col_btn_fill = st.columns([1.2, 1, 4])
        with col_btn1:
            submit_pressed = st.button("🚀 Get Answer", type="primary", use_container_width=True, disabled=not query.strip())
        with col_btn2:
            if st.button("🧹 Clear", use_container_width=True):
                st.session_state.query_input = ""
                st.rerun()

        if submit_pressed and query.strip():
            with st.spinner("Searching relevant chunks and composing answer..."):
                hits = retrieve(query, chunks, top_k=top_k)
                
                if not hits or hits[0]["score"] <= 0:
                    st.warning("⚠️ **No relevant passages found.** Try phrasing your question differently or ensure the uploaded document contains information regarding this topic.")
                else:
                    answer = answer_from_chunks(query, hits, max_sentences=max_sentences)
                    top_score = hits[0]["score"]
                    
                    # Store in search history
                    st.session_state.history.insert(0, {
                        "query": query,
                        "answer": answer,
                        "top_score": top_score,
                        "hits_count": len(hits)
                    })
                    
                    # Score badge category
                    if top_score >= 0.5:
                        badge_html = f'<span class="badge-pill badge-high">High Confidence ({top_score:.3f})</span>'
                    elif top_score >= 0.25:
                        badge_html = f'<span class="badge-pill badge-mid">Moderate Relevance ({top_score:.3f})</span>'
                    else:
                        badge_html = f'<span class="badge-pill badge-low">Weak Match ({top_score:.3f})</span>'

                    # Display Answer Box
                    st.markdown("---")
                    st.markdown(
                        f"""
                        <div class="answer-box">
                            <div style="margin-bottom: 8px;">
                                <strong>💡 Synthesized Extractive Answer</strong> &nbsp;{badge_html}
                            </div>
                            <div style="font-size: 1.05rem; color: inherit;">
                                {answer}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Metric summaries for retrieval
                    c_m1, c_m2, c_m3 = st.columns(3)
                    c_m1.metric("Top Passage Score", f"{top_score:.3f}")
                    c_m2.metric("Passages Retrieved", f"{len(hits)} of {len(chunks)}")
                    c_m3.metric("Search Coverage", f"{min(100, int(top_score * 120))}%")
                    
                    # Retrieved Sources Breakdown
                    st.markdown("#### 📖 Retrieved Sources & Evidence")
                    st.caption("Inspect the exact passages extracted from your corpus to verify grounding.")
                    
                    for i, hit in enumerate(hits, 1):
                        score = hit["score"]
                        score_pct = min(1.0, max(0.0, score))
                        score_label = "High" if score >= 0.5 else ("Medium" if score >= 0.25 else "Low")
                        
                        header_title = f"Excerpt {i} · {hit['source']} (Chunk #{hit['chunk_id']}) — Match Score: {score:.3f}"
                        with st.expander(header_title, expanded=(i == 1)):
                            st.progress(score_pct, text=f"Relevance: {score_label} ({score:.3f})")
                            st.markdown(f"> *\"{hit['text']}\"*")
                            st.caption(f"Source file: `{hit['source']}` | Chunk ID: `{hit['chunk_id']}`")

        # History section
        if st.session_state.history:
            with st.expander(f"🕒 Recent Questions in this Session ({len(st.session_state.history)})", expanded=False):
                for idx, item in enumerate(st.session_state.history[:5], 1):
                    st.markdown(f"**Q{idx}:** {item['query']}")
                    st.markdown(f"*{item['answer'][:180]}...*")
                    st.caption(f"Top score: {item['top_score']:.3f} | {item['hits_count']} citations")
                    st.divider()

# -------------------------------------------------------------
# TAB 2: CORPUS & CHUNKS INSPECTOR
# -------------------------------------------------------------
with tab_inspector:
    st.markdown("#### 🗂️ Indexed Documents & Chunk Inspector")
    if not chunks:
        st.info("Upload documents from the sidebar to inspect word counts, chunk boundaries, and token distribution.")
    else:
        # Overview Cards
        tot_words = sum(stats.get("word_count", 0) for stats in file_stats.values())
        tot_chars = sum(stats.get("char_count", 0) for stats in file_stats.values())
        
        stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
        stat_c1.metric("Indexed Documents", len(items))
        stat_c2.metric("Total Generated Chunks", len(chunks))
        stat_c3.metric("Total Words Extracted", f"{tot_words:,}")
        stat_c4.metric("Avg Words per Chunk", f"{tot_words // max(1, len(chunks))}")
        
        st.markdown("##### 📄 Document Breakdown")
        doc_data = []
        for doc_name, stat in file_stats.items():
            if "error" in stat:
                doc_data.append({"Document": doc_name, "Status": f"❌ Error: {stat['error']}", "Words": 0, "Chunks": 0})
            else:
                doc_data.append({
                    "Document": doc_name,
                    "Status": "✅ Successfully Indexed",
                    "Words": f"{stat['word_count']:,}",
                    "Chunks": stat['chunk_count'],
                    "Characters": f"{stat['char_count']:,}"
                })
        st.dataframe(pd.DataFrame(doc_data), use_container_width=True)
        
        st.markdown("---")
        st.markdown("##### 🔎 Chunk Explorer")
        filter_doc = st.selectbox("Select document to filter chunks:", ["All Documents"] + [f[0] for f in items])
        
        displayed_chunks = chunks if filter_doc == "All Documents" else [c for c in chunks if c["source"] == filter_doc]
        
        chunk_select_idx = st.slider("Select chunk index to preview:", 1, len(displayed_chunks), 1, step=1)
        selected_chunk = displayed_chunks[chunk_select_idx - 1]
        
        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 15px;">
                <strong>Document:</strong> <code>{selected_chunk['source']}</code> &nbsp;|&nbsp;
                <strong>Chunk ID:</strong> <code>{selected_chunk['chunk_id']}</code> &nbsp;|&nbsp;
                <strong>Length:</strong> <code>{len(selected_chunk['text'].split())} words</code>
                <hr style="margin: 10px 0; border: none; border-top: 1px solid rgba(255,255,255,0.08);"/>
                <div style="white-space: pre-wrap; font-family: monospace; font-size: 0.95rem;">{selected_chunk['text']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# -------------------------------------------------------------
# TAB 3: DATASET EVALUATION & INSIGHTS
# -------------------------------------------------------------
with tab_dataset:
    st.markdown("#### 📊 Evaluation Dataset Analysis")
    st.caption("Descriptive exploratory analysis of the supplied `data/dataset.csv` benchmark metadata.")
    
    dataset_path = Path(__file__).parent / 'data' / 'dataset.csv'
    if not dataset_path.exists():
        st.error(f"Dataset file not found at `{dataset_path}`.")
    else:
        try:
            df = pd.read_csv(dataset_path)
            
            # Key KPI metrics
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Evaluation Rows", f"{len(df):,}")
            kpi2.metric("Unique Documents", f"{df['doc_id'].nunique():,}")
            kpi3.metric("Mean Retrieval Score", f"{df['retrieval_score'].mean():.3f}")
            kpi4.metric("Mean Answer Quality", f"{df['answer_quality_score'].mean():.3f}")
            
            st.markdown("---")
            
            # Interactive Filter
            doc_types = ["All"] + sorted(df['doc_type'].dropna().unique().tolist())
            selected_type = st.selectbox("Filter analysis by document type:", doc_types)
            
            filtered_df = df if selected_type == "All" else df[df['doc_type'] == selected_type]
            
            # Matplotlib Visualizations with Polished Theme
            plt.style.use('dark_background' if st.get_option("theme.base") == "dark" else 'default')
            
            col_chart1, col_chart2 = st.columns(2)
            
            # Chart 1: Rows by Document Type
            with col_chart1:
                fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=150)
                type_counts = df.groupby('doc_type').size().sort_values(ascending=True)
                bars = ax1.barh(type_counts.index, type_counts.values, color='#3B82F6', height=0.6, edgecolor='#1D4ED8')
                ax1.set_title('Evaluation Rows by Document Type', fontsize=12, fontweight='bold', pad=12)
                ax1.set_xlabel('Row Count', fontsize=10)
                ax1.set_ylabel('Document Type', fontsize=10)
                ax1.grid(axis='x', linestyle='--', alpha=0.3)
                for bar in bars:
                    w = bar.get_width()
                    ax1.text(w + 2, bar.get_y() + bar.get_height()/2, f'{int(w)}', va='center', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig1, clear_figure=True)
            
            # Chart 2: Mean Answer Quality by Query Type
            with col_chart2:
                fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=150)
                mean_quality = filtered_df.groupby('query_type')['answer_quality_score'].mean().sort_values(ascending=True)
                bars2 = ax2.barh(mean_quality.index, mean_quality.values, color='#10B981', height=0.6, edgecolor='#047857')
                ax2.set_title(f'Mean Answer Quality by Query Type ({selected_type})', fontsize=12, fontweight='bold', pad=12)
                ax2.set_xlabel('Answer Quality Score (0 to 1)', fontsize=10)
                ax2.set_ylabel('Query Type', fontsize=10)
                ax2.grid(axis='x', linestyle='--', alpha=0.3)
                for bar in bars2:
                    w = bar.get_width()
                    ax2.text(w + 0.01, bar.get_y() + bar.get_height()/2, f'{w:.3f}', va='center', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig2, clear_figure=True)
                
            # Chart 3: Correlation Scatter Plot
            st.markdown("##### 📈 Retrieval Score vs. Answer Quality Score")
            fig3, ax3 = plt.subplots(figsize=(10, 4), dpi=150)
            scatter = ax3.scatter(
                filtered_df['retrieval_score'],
                filtered_df['answer_quality_score'],
                c=filtered_df['word_count'],
                cmap='viridis',
                alpha=0.65,
                s=28,
                edgecolors='none'
            )
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('Chunk Word Count', fontsize=9)
            ax3.set_title('Correlation Analysis: Retrieval Precision vs. Answer Quality', fontsize=12, fontweight='bold', pad=12)
            ax3.set_xlabel('Retrieval Score (Lexical Similarity)', fontsize=10)
            ax3.set_ylabel('Answer Quality Score', fontsize=10)
            ax3.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig3, clear_figure=True)
            
            # Interactive Table Preview
            with st.expander("📋 Browse Raw Evaluation Data (500 Records)", expanded=False):
                st.dataframe(filtered_df, use_container_width=True)
                csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Filtered Dataset CSV",
                    data=csv_bytes,
                    file_name="rag_evaluation_filtered.csv",
                    mime="text/csv"
                )
                
        except Exception as ex:
            st.error(f"Failed to load dataset: {ex}")

# -------------------------------------------------------------
# TAB 4: ARCHITECTURE & PIPELINE
# -------------------------------------------------------------
with tab_about:
    st.markdown("#### 🏗️ Architecture & Retrieval Mechanics")
    st.markdown(
        """
        DocuRAG implements a **fully transparent, deterministic lexical RAG pipeline** tailored for privacy, offline speed, and zero cloud API dependencies.
        
        ##### 🔄 System Pipeline
        """
    )
    st.code(
        """
[User Document: PDF / Word / TXT / MD]
                 │
                 ▼
      [Document Text Parser]
   (pypdf / python-docx / utf-8)
                 │
                 ▼
       [Windowed Chunking]
    (Sliding window with overlap)
                 │
                 ▼
  [Lexical Inverted Index & Token Scorer] ◄── [User Question Query]
                 │
                 ▼
      [Top-K Passage Retrieval]
    (Overlap Coverage + Density)
                 │
                 ▼
    [Extractive Sentence Synthesis]
                 │
                 ▼
   [Answer Grounded with Citations]
        """,
        language="text"
    )
    
    st.markdown(
        """
        ##### ⚙️ Algorithmic Mechanics:
        1. **Multi-Format Ingestion**: Extracts raw text from `.pdf`, `.docx`, `.txt`, and `.md` using in-memory byte buffers without saving files to disk.
        2. **Windowed Chunking**: Splits textual data into $N$-word blocks with sliding overlap (default: 180 words with 35 words overlap) to ensure boundary continuity across paragraphs.
        3. **Normalized Token Overlap Ranking**: Calculates frequency-adjusted matching score combining:
           - *Token Coverage* ($85\\%$ weight): fraction of unique query terms matched.
           - *Token Density* ($15\\%$ weight): density of matches within the target chunk.
        4. **Extractive Grounding**: Synthesizes responses strictly from matching sentences in retrieved chunks, eliminating hallucinations.
        """
    )
    st.divider()
    st.caption("Developed by MD Taslim | RAG-Based Document Analyser & Question Answering System")
