# RAG-Based Document Q&A System

**A local, source-grounded document question-answering prototype built with Python and Streamlit.**

Upload PDF, Word, text, or Markdown documents, ask a question, and inspect the passages used to form the answer. The app also includes an exploratory dashboard for the project’s supplied dataset.

**Author:** MD Taslim  
**Domain:** LLM / Python  
**Status:** Working prototype

## Highlights

- Upload one or more PDF, DOCX, TXT, and Markdown files.
- Extract and split document text into overlapping chunks.
- Rank chunks with a lightweight token-overlap retrieval baseline.
- Present a concise extractive answer with the retrieved source excerpts.
- Explore the supplied CSV through summary metrics and three visualizations.
- Run locally without API credentials or a hosted language model.

## Run locally

### Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer
- Internet access on first run to install the Python packages

### Windows quick start

Double-click `run.bat`. It creates a virtual environment, installs the packages, and starts the app.

Or use PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Upload a document in the sidebar, enter a question, and select **Get answer**. Uploaded files are processed in memory by the app and are not written into the project folder.

## How it works

```text
PDF / DOCX / TXT / Markdown
              │
              ▼
        Text extraction
              │
              ▼
     Overlapping text chunks
              │
              ▼
   Token-overlap chunk ranking ◄── User question
              │
              ▼
 Extractive answer + source excerpts
```

The retrieval score is a simple lexical relevance heuristic. It is not an embedding similarity, confidence estimate, or calibrated probability. The answer is composed from matching sentences in the retrieved text; this prototype does not generate new content with an LLM. Scanned PDFs need OCR before upload.

## Supplied dataset

The provided `data/dataset.csv` contains 500 rows and seven columns: `doc_id`, `doc_type`, `chunk_id`, `word_count`, `retrieval_score`, `query_type`, and `answer_quality_score`. The dashboard summarizes these values and plots:

1. Row counts by document type
2. Mean answer-quality score by query type
3. Retrieval score against answer-quality score

The CSV does **not** include document text, user queries, or generated answers. It therefore supports descriptive analysis, but cannot serve as the document corpus or as a direct end-to-end Q&A evaluation set. No synthetic Q&A rows are presented as real evaluation data. The copied data dictionary is at `data/data_dictionary.csv`.

## Project structure

```text
rag-document-qa/
├── app.py                    # Streamlit interface and dataset dashboard
├── rag_core.py               # Parsing, chunking, ranking, answer composition
├── requirements.txt          # Python dependencies
├── run.bat                   # Windows setup and launch helper
└── data/
    ├── dataset.csv           # Supplied 500-row dataset
    └── data_dictionary.csv   # Column descriptions
```

## Limitations and responsible use

- Review the displayed source excerpts; retrieval can miss relevant passages.
- Answers may be incomplete or irrelevant when documents do not contain the requested information.
- Do not use this prototype for legal or medical advice without expert review.
- Do not upload confidential documents to an environment you do not control.

## Technology

Python · Streamlit · pandas · Matplotlib · pypdf · python-docx

## Author

**MD Taslim**
