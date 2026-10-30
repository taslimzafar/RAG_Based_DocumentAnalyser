"""Small dependency-light document parsing, chunking, and lexical retrieval."""
import re
from pathlib import Path
from collections import Counter

WORD_RE = re.compile(r"[a-zA-Z0-9]+(?:['’-][a-zA-Z0-9]+)?")
STOP = set("a an and are as at be been being by can could did do does for from had has have how i in is it its may might of on or our shall should than that the their them there these they this those to was we were what when where which who why will with would you your".split())

def parse_upload(name: str, data: bytes) -> str:
    suffix=Path(name).suffix.lower()
    if suffix in {'.txt','.md'}:
        return data.decode('utf-8-sig',errors='replace')
    if suffix=='.pdf':
        from pypdf import PdfReader
        import io
        reader=PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or '' for page in reader.pages)
    if suffix=='.docx':
        from docx import Document
        import io
        doc=Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    raise ValueError(f"Unsupported file type: {suffix}")

def chunk_text(text: str, source: str, size: int=180, overlap: int=35):
    words=text.split()
    chunks=[]
    if not words:
        return chunks
    step=max(1,size-overlap)
    for start in range(0,len(words),step):
        body=" ".join(words[start:start+size]).strip()
        if body:
            chunks.append({"source":source,"chunk_id":len(chunks)+1,"text":body})
        if start+size>=len(words):
            break
    return chunks

def _tokens(text):
    return [w.lower() for w in WORD_RE.findall(text) if w.lower() not in STOP]

def retrieve(query: str, chunks: list, top_k: int=4):
    q=Counter(_tokens(query))
    if not q:
        return []
    scored=[]
    for chunk in chunks:
        terms=Counter(_tokens(chunk['text']))
        overlap=sum(min(count,terms[token]) for token,count in q.items())
        coverage=overlap/max(1,sum(q.values()))
        density=overlap/(max(1,sum(terms.values()))**0.35)
        score=coverage*0.85+density*0.15
        if score>0:
            scored.append({**chunk,"score":score})
    return sorted(scored,key=lambda x:x['score'],reverse=True)[:top_k]

def answer_from_chunks(question: str, hits: list, max_sentences: int=5) -> str:
    q=set(_tokens(question))
    candidates=[]
    for hit in hits:
        for sentence in re.split(r'(?<=[.!?])\s+',hit['text']):
            words=set(_tokens(sentence))
            overlap=len(q & words)
            if sentence.strip() and overlap:
                candidates.append((overlap/max(1,len(q)),sentence.strip()))
    selected=[]
    seen=set()
    for _, sentence in sorted(candidates,key=lambda x:x[0],reverse=True):
        key=sentence.lower()
        if key not in seen:
            selected.append(sentence)
            seen.add(key)
        if len(selected)>=max_sentences:
            break
    return " ".join(selected) if selected else "The retrieved passages do not contain enough matching text to form an answer."
