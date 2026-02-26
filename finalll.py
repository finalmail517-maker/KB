import os
import io
import json
import hashlib
import logging
import subprocess
from typing import List, Dict, Any, Tuple
import numpy as np
import streamlit as st
from dotenv import load_dotenv
# transcription
import yt_dlp
import whisper
# embeddings / LLM / FAISS
from huggingface_hub import login as hf_login
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationEntityMemory
from sentence_transformers import SentenceTransformer, util
import os
import streamlit as st
from typing import List, Dict
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
# ===================== ENV & CONSTANTS =====================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
DATA_DIR = "data"
INDEX_DIR_PDF = "faiss_pdf"
INDEX_DIR_VIDEO = "faiss_video"
INDEX_DIR_COMBINED = "faiss_combined"
META_FILE = os.path.join(DATA_DIR, "index_meta.json")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR_PDF, exist_ok=True)
os.makedirs(INDEX_DIR_VIDEO, exist_ok=True)
os.makedirs(INDEX_DIR_COMBINED, exist_ok=True)
# ===================== LOGGING =====================
logger = logging.getLogger("smart_qa")
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    handler = logging.FileHandler("smart_qa.log")
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
# ===================== HUGGING FACE LOGIN (OPTIONAL) =====================
if HUGGINGFACE_API_KEY:
    try:
        hf_login(token=HUGGINGFACE_API_KEY)
    except Exception as e:
        logger.warning("HuggingFace login failed (continuing without): %s", e)
# ===================== RERANKER =====================
use_crossencoder = True
try:
    RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception as e:
    logger.warning("CrossEncoder unavailable (%s). Falling back to cosine reranker.", e)
    use_crossencoder = False
    RERANKER = None
# ===================== WHISPER =====================
try:
    WHISPER_MODEL = whisper.load_model("base")
except Exception as e:
    logger.warning("Failed to load Whisper 'base' (%s). Trying 'tiny'...", e)
    WHISPER_MODEL = whisper.load_model("tiny")
# ===================== UTILS =====================
def seconds_to_hhmmss(seconds: float) -> str:
    sec = int(round(seconds))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
def url_to_basename(url: str) -> str:
    return "transcript_" + hashlib.md5(url.encode()).hexdigest()[:10]
def url_to_json_path(url: str) -> str:
    return os.path.join(DATA_DIR, url_to_basename(url) + ".json")
def load_meta() -> Dict[str, Any]:
    """Persistent registry of files added and indexed."""
    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            logger.warning("Failed to read META_FILE: %s", e)
            meta = {}
    else:
        meta = {}
    # normalize keys
    meta.setdefault("pdfs", [])
    meta.setdefault("videos", [])
    meta.setdefault("indexed_pdfs", [])
    meta.setdefault("indexed_videos", [])
    meta.setdefault("indexed_combined", [])
    return meta
def save_meta(meta: Dict[str, Any]) -> None:
    try:
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("Failed to write META_FILE: %s", e)

def is_video_only_mode(pdf_files: List[str], video_jsons: List[str]) -> bool:
    """Returns True if only video files are loaded (no PDFs/TXT)."""
    pdf_count = len([f for f in pdf_files if os.path.exists(f)])
    print(pdf_count)
    return pdf_count == 0 and len(video_jsons) > 0
# ===================== TRANSCRIPTION =====================
def transcribe_url(url: str) -> str:
    """
    Download + transcribe a video to JSON once, then reuse.
    JSON fields: video_url, text, segments[{start,end,text}]
    """
    json_path = url_to_json_path(url)
    if os.path.exists(json_path):
        return json_path
    ydl_opts = {"quiet": True, "format": "bestaudio/best", "dump_single_json": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        audio_url = info.get("url") or url
    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        audio_url,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-f",
        "s16le",
        "-"
    ]
    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    audio_bytes, _ = process.communicate()
    audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
    result = WHISPER_MODEL.transcribe(audio_np)
    segments = result.get("segments") or []
    structured = {"video_url": url, "text": result.get("text", ""), "segments": segments}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)
    return json_path
# ===================== LOADING & CHUNKING (with chunk_index) =====================
def load_and_split_document(path: str) -> List[Document]:
    """
    Returns a list of LangChain Documents with metadata:
      - PDF/TXT: source=filename, source_type='pdf', chunk_index
      - Transcript JSON: source=video_url, source_type='video', start_time, end_time, chunk_index
    """
    try:
        docs: List[Document] = []
        if path.endswith(".json"):  # transcript
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            video_url = data.get("video_url", "")
            segments = data.get("segments", [])
            # group segments into ~1000-char chunks with overlapping last segment
            chunk_size = 1000
            buf, cur_len = [], 0
            cur_start, cur_end = None, None
            chunk_idx = 0
            for seg in segments:
                t = (seg.get("text") or "").strip()
                if not t:
                    continue
                s_start = float(seg.get("start", 0.0))
                s_end = float(seg.get("end", s_start))
                if cur_start is None:
                    cur_start = s_start
                buf.append(t)
                cur_end = s_end
                cur_len += len(t)
                if cur_len >= chunk_size:
                    content = "\n".join(buf).strip()
                    if content:
                        docs.append(Document(
                            page_content=content,
                            metadata={
                                "source": video_url,
                                "source_type": "video",
                                "start_time": cur_start,
                                "end_time": cur_end,
                                "chunk_index": chunk_idx
                            }
                        ))
                        chunk_idx += 1
                    # overlap one last segment
                    last = buf[-1:]
                    buf = list(last)
                    cur_len = sum(len(x) for x in buf)
                    # keep same start
            if buf:
                content = "\n".join(buf).strip()
                if content:
                    docs.append(Document(
                        page_content=content,
                        metadata={
                            "source": video_url,
                            "source_type": "video",
                            "start_time": cur_start if cur_start is not None else 0.0,
                            "end_time": cur_end if cur_end is not None else 0.0,
                            "chunk_index": chunk_idx
                        }
                    ))
            return docs
        # pdf/txt
        if path.lower().endswith(".pdf"):
            from PyPDF2 import PdfReader
            reader = PdfReader(path)
            text = "\n".join([p.extract_text() or "" for p in reader.pages])
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        split_docs = splitter.create_documents([text], metadatas=[{"source": os.path.basename(path), "source_type": "pdf"}])
        # add chunk_index metadata so we can merge neighbors later
        docs: List[Document] = []
        for i, d in enumerate(split_docs):
            md = dict(d.metadata or {})
            md["chunk_index"] = i
            docs.append(Document(page_content=d.page_content, metadata=md))
        return docs
    except Exception as e:
        logger.error("Error reading %s: %s", path, e)
        return []
# ===================== LLM & EMBEDDINGS =====================
@st.cache_resource
def init_llm_and_embeddings():
    # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=GEMINI_API_KEY)
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=GEMINI_API_KEY,
    temperature=0.0,      # Make output more deterministic
    top_p=0.95,            # Consider all tokens
    # top_k=40          # Pick most likely next token
   
)
    try:
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    except Exception as e:
        logger.warning("HF embeddings unavailable (%s). Falling back to Google embeddings.", e)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    return llm, embeddings
# ===================== FAISS SAVE/LOAD HELPERS (save doc_store for contextual merging) =====================
def build_index_from_files(files: List[str], embeddings, index_dir: str) -> Tuple[FAISS, Dict[str, List[Dict[str, Any]]]]:
    """
    Build FAISS index from files and also persist an ordered doc-store that maps:
      source -> list of {'content': ..., 'metadata': {...}}
    The doc_store is used for contextual merging of neighbor chunks at query time.
    Returns tuple (faiss_vectorstore, doc_store)
    """
    docs = []
    doc_store: Dict[str, List[Dict[str, Any]]] = {}
    for f in files:
        file_docs = load_and_split_document(f)
        # append docs to main list and populate doc_store preserving order per source
        for d in file_docs:
            docs.append(d)
            src = (d.metadata or {}).get("source", os.path.basename(f))
            if src not in doc_store:
                doc_store[src] = []
            # store both content and metadata so we can reconstruct neighbors
            doc_store[src].append({
                "content": d.page_content,
                "metadata": d.metadata or {}
            })
    if not docs:
        return None, {}
    vs = FAISS.from_documents(docs, embeddings)
    # ensure index_dir exists
    os.makedirs(index_dir, exist_ok=True)
    vs.save_local(index_dir)
    # save doc_store as JSON for this index
    try:
        with open(os.path.join(index_dir, "doc_store.json"), "w", encoding="utf-8") as f:
            json.dump(doc_store, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("Failed to save doc_store for %s: %s", index_dir, e)
    return vs, doc_store
def load_index(index_dir: str, embeddings) -> Tuple[Any, Dict[str, List[Dict[str, Any]]]]:
    """
    Returns tuple (faiss_vectorstore or None, doc_store dict or {}).
    If FAISS index exists on disk, load it. Also attempt to load doc_store.json.
    """
    doc_store = {}
    if os.path.exists(index_dir):
        try:
            vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
            # try to load doc_store
            ds_path = os.path.join(index_dir, "doc_store.json")
            if os.path.exists(ds_path):
                try:
                    with open(ds_path, "r", encoding="utf-8") as f:
                        doc_store = json.load(f)
                except Exception as e:
                    logger.warning("Failed to load doc_store at %s: %s", ds_path, e)
            return vs, doc_store
        except Exception as e:
            logger.warning("Failed to load index at %s: %s", index_dir, e)
    return None, {}
# === NEW: INCREMENTAL UPDATE HELPERS ==========================================
def _expected_source_for_file(path: str) -> str:
    """Return the 'source' key that load_and_split_document will use for this path."""
    if path.endswith(".json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("video_url", "") or os.path.basename(path)
        except Exception:
            return os.path.basename(path)
    # pdf/txt
    return os.path.basename(path)
def update_index_incremental(files: List[str], embeddings, index_dir: str) -> Tuple[Any, Dict[str, List[Dict[str, Any]]], List[str]]:
    """
    Append only NEW files (by source id) to an existing FAISS + doc_store.
    Returns (vs, doc_store, added_sources)
    """
    vs_existing, doc_store = load_index(index_dir, embeddings)
    existing_sources = set(doc_store.keys()) if doc_store else set()
    # choose files whose expected 'source' not yet in doc_store
    new_files = []
    new_sources = []
    for f in files:
        src = _expected_source_for_file(f)
        if src and src not in existing_sources:
            new_files.append(f)
            new_sources.append(src)
    if not new_files:
        # nothing new; return existing as-is (or empty if none)
        return vs_existing, (doc_store or {}), []
    # build docs for new files only
    docs = []
    # also extend doc_store incrementally
    if doc_store is None:
        doc_store = {}
    for f in new_files:
        file_docs = load_and_split_document(f)
        for d in file_docs:
            docs.append(d)
            src = (d.metadata or {}).get("source", os.path.basename(f))
            if src not in doc_store:
                doc_store[src] = []
            doc_store[src].append({
                "content": d.page_content,
                "metadata": d.metadata or {}
            })
    if not docs:
        return vs_existing, doc_store, []
    vs_new = FAISS.from_documents(docs, embeddings)
    if vs_existing is None:
        vs_existing = vs_new
    else:
        # merge new vectors into existing
        vs_existing.merge_from(vs_new)
    # persist
    os.makedirs(index_dir, exist_ok=True)
    vs_existing.save_local(index_dir)
    try:
        with open(os.path.join(index_dir, "doc_store.json"), "w", encoding="utf-8") as f:
            json.dump(doc_store, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("Failed to save updated doc_store for %s: %s", index_dir, e)
    return vs_existing, doc_store, new_sources
# ===================== RERANKERS =====================
def cosine_rerank(query: str, docs: List[Document], embed_model, top_k=5) -> List[Document]:
    if not docs:
        return []
    q = embed_model.embed_query(query)
    embs = embed_model.embed_documents([d.page_content for d in docs])
    sims = cosine_similarity([q], embs)[0]
    ranking = sorted(zip(docs, sims), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranking[:top_k]]
def crossencode_rerank(query: str, docs: List[Document], top_k=5) -> List[Document]:
    pairs = [[query, d.page_content] for d in docs]
    scores = RERANKER.predict(pairs)
    ranking = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranking[:top_k]]
# ===================== CONTEXTUAL MERGING HELPERS =====================
def expand_with_neighbors(sel_docs: List[Document], doc_store: Dict[str, List[Dict[str, Any]]], window: int = 1) -> List[Document]:
    """
    For each selected Document, expand by merging its neighbor chunks from doc_store.
    window = how many neighbors to include on each side (e.g. 1 -> include idx-1, idx, idx+1)
    Returns a deduped list of Documents (merged content per source).
    """
    if not sel_docs or not doc_store:
        return sel_docs
    merged_texts = []  # keep (source, merged_content, metadata)
    seen = set()
    for d in sel_docs:
        md = d.metadata or {}
        src = md.get("source", "")
        if not src:
            key = (src, d.page_content[:200])
            if key in seen:
                continue
            seen.add(key)
            merged_texts.append((src, d.page_content, md))
            continue
        src_list = doc_store.get(src, [])
        if not src_list:
            key = (src, d.page_content[:200])
            if key in seen:
                continue
            seen.add(key)
            merged_texts.append((src, d.page_content, md))
            continue
        chunk_idx = md.get("chunk_index")
        if chunk_idx is None:
            if md.get("source_type") == "video":
                start_time = md.get("start_time", None)
                found_idx = None
                for i, item in enumerate(src_list):
                    imd = item.get("metadata", {})
                    if "start_time" in imd and float(imd.get("start_time", 0.0)) == float(start_time or 0.0):
                        found_idx = i
                        break
                chunk_idx = found_idx
            if chunk_idx is None:
                found_idx = None
                for i, item in enumerate(src_list):
                    if item.get("content", "").strip() == d.page_content.strip():
                        found_idx = i
                        break
                chunk_idx = found_idx
        if chunk_idx is not None and isinstance(chunk_idx, int):
            start = max(0, chunk_idx - window)
            end = min(len(src_list) - 1, chunk_idx + window)
            parts = [src_list[i]["content"] for i in range(start, end + 1)]
            merged_content = "\n".join(parts).strip()
            key = (src, start, end, merged_content[:200])
            if key in seen:
                continue
            seen.add(key)
            merged_md = {}
            try:
                m_start = None
                m_end = None
                for i in range(start, end + 1):
                    imd = src_list[i].get("metadata", {}) or {}
                    if "start_time" in imd:
                        stt = float(imd.get("start_time", 0.0) or 0.0)
                        if m_start is None or stt < m_start:
                            m_start = stt
                    if "end_time" in imd:
                        edt = float(imd.get("end_time", 0.0) or 0.0)
                        if m_end is None or edt > m_end:
                            m_end = edt
                if m_start is not None:
                    merged_md["start_time"] = m_start
                if m_end is not None:
                    merged_md["end_time"] = m_end
                merged_md["source"] = src
                if "source_type" in md:
                    merged_md["source_type"] = md.get("source_type")
                merged_md["chunk_index"] = start
            except Exception:
                merged_md["source"] = src
            merged_texts.append((src, merged_content, merged_md))
        else:
            key = (src, d.page_content[:200])
            if key in seen:
                continue
            seen.add(key)
            merged_texts.append((src, d.page_content, md))
    merged_docs = []
    for src, content, md in merged_texts:
        merged_docs.append(Document(page_content=content, metadata=md))
    return merged_docs
from sentence_transformers import SentenceTransformer, util
import os
import streamlit as st
from typing import List, Dict
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
# =========================
# GLOBAL QA MEMORY
# =========================
video_QA_index = None   # long-term memory (video Q&A stored in FAISS)
kb_QA_index = None      # long-term memory (KB Q&A stored in FAISS)
# directories for saving indexes
QA_INDEX_DIR = "qa_indexes"
VIDEO_QA_PATH = os.path.join(QA_INDEX_DIR, "video_QA_index")
KB_QA_PATH = os.path.join(QA_INDEX_DIR, "kb_QA_index")
os.makedirs(QA_INDEX_DIR, exist_ok=True)
# =========================
# SAVE & LOAD QA INDEXES
# =========================
def save_QA_indexes():
    """Persist long-term memory indexes (FAISS) to disk."""
    global video_QA_index, kb_QA_index
    if video_QA_index:
        video_QA_index.save_local(VIDEO_QA_PATH)
    if kb_QA_index:
        kb_QA_index.save_local(KB_QA_PATH)
def load_QA_indexes(embeddings):
    """Load long-term memory indexes from disk."""
    global video_QA_index, kb_QA_index
    if os.path.exists(VIDEO_QA_PATH):
        try:
            video_QA_index = FAISS.load_local(VIDEO_QA_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print("Could not load video_QA_index:", e)
    if os.path.exists(KB_QA_PATH):
        try:
            kb_QA_index = FAISS.load_local(KB_QA_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print("Could not load kb_QA_index:", e)

# =========================
# ADD TO QA MEMORY (FAISS)
# =========================
def add_to_QA_index(query: str, answer: str, source_type: str, embeddings):
    """Add a Q&A pair to long-term memory (FAISS index)."""
    global video_QA_index, kb_QA_index
    if not answer or answer.strip() == "The answer is not present in the documents.":
        return  # skip empty answers
    doc = Document(page_content=f"Q: {query}\nA: {answer}", metadata={"source_type": source_type})
    if source_type == "video":
        if video_QA_index is None:
            video_QA_index = FAISS.from_documents([doc], embeddings)
        else:
            video_QA_index.add_documents([doc])
    elif source_type == "pdf":
        if kb_QA_index is None:
            kb_QA_index = FAISS.from_documents([doc], embeddings)
        else:
            kb_QA_index.add_documents([doc])
    save_QA_indexes()
# Load embeddings model once (lightweight, e.g. all-MiniLM-L6-v2)
# _elab_model = SentenceTransformer("all-MiniLM-L6-v2")
_elab_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
# ===================== ELABORATION DETECTION =====================
def _is_elaboration(query: str, last_answer: str, threshold: float = 0.65) -> bool:
    if not last_answer.strip():
        return False
    q_emb = _elab_model.encode(query, convert_to_tensor=True)
    a_emb = _elab_model.encode(last_answer, convert_to_tensor=True)
    sim = util.cos_sim(q_emb, a_emb).item()
    return sim >= threshold
def get_memory_context(query: str) -> str:
    """Retrieve conversation context from ConversationEntityMemory."""
    return st.session_state.entity_memory.load_memory_variables({"input": query})["history"]
from difflib import SequenceMatcher
from langchain.prompts import PromptTemplate
# ```python
def answer_from_docs(query: str, docs: List[Document], llm, source_type: str) -> Dict[str, str]:
    """
    Generates long, structured answers using video and/or KB context.
    - Decides which prompt to use strictly based on document source_type metadata.
    - If only video content is relevant → both video and KB answers are identical.
    - If both video + KB exist → merges them naturally.
    - Produces detailed, educational, tutorial-style responses.
    """

    # --- Memory Setup ---
    try:
        memory_context = st.session_state.entity_memory.load_memory_variables({"input": query}).get("history", "")
    except Exception:
        memory_context = ""
    last_question, last_answer = "", ""
    if hasattr(st.session_state, "entity_memory") and st.session_state.entity_memory.buffer:
        buf = st.session_state.entity_memory.buffer
        if isinstance(buf, str):
            history = buf.split("\n")
            if len(history) >= 2:
                last_question = history[-2].replace("Human:", "").strip()
                last_answer = history[-1].replace("AI:", "").strip()
        elif isinstance(buf, list):
            last_turn = buf[-1]
            if isinstance(last_turn, dict):
                last_question = last_turn.get("input", "")
                last_answer = last_turn.get("output", "")
            elif isinstance(last_turn, (list, tuple)) and len(last_turn) == 2:
                last_question, last_answer = last_turn

    # =========================================================
    # ELABORATION DETECTION
    # =========================================================
    elaboration_mode = _is_elaboration(query, last_answer)
    if elaboration_mode and last_question and last_answer:
        query = f"{query} (follow-up to: Q: {last_question}, A: {last_answer})"
    global video_QA_index, kb_QA_index
    video_mem, kb_mem = "", ""
    try:
        if video_QA_index:
            vid = video_QA_index.similarity_search(query, k=1)
            if vid:
                video_mem = vid[0].page_content
    except Exception:
        pass
    try:
        if kb_QA_index:
            kb = kb_QA_index.similarity_search(query, k=1)
            if kb:
                kb_mem = kb[0].page_content
    except Exception:
        pass

    # --- Split Context by Source Type ---
    video_context = "\n".join([d.page_content for d in docs if (d.metadata or {}).get("source_type") == "video"]).strip()
    kb_context = "\n".join([d.page_content for d in docs if (d.metadata or {}).get("source_type") == "pdf"]).strip()

    # --- Prompts ---
    PROMPT_VIDEO_ONLY = PromptTemplate.from_template("""
You are a helpful and knowledgeable assistant answering questions based ONLY on the provided video transcript.
User question: {question}
Short-term memory (recent conversation):
{memory_context}
Long-term memory (previously answered Q&A from video):
{long_term}
Video context (your ONLY source of truth):
{video_context}
Instructions:
- Provide a **long, detailed, and tutorial-style** explanation that fully answers the question.
- Use **the same tone and phrasing** from the video — include examples, analogies, and context.
- Do NOT add outside knowledge or mention documents.
- If no relevant info exists, reply EXACTLY: "The answer is not present in the videos."
""")

    PROMPT_KB_ONLY = PromptTemplate.from_template("""
You are a helpful and knowledgeable assistant answering questions based ONLY on the provided knowledge base (PDF/text documents).
User question: {question}
Short-term memory (recent conversation):
{memory_context}
Long-term memory (previously answered Q&A from KB):
{long_term}
Knowledge Base context (your ONLY source of truth):
{kb_context}
Instructions:
- Write a **comprehensive, well-organized, and factual** explanation using only KB data.
- Include examples or applications if mentioned.
- Do NOT refer to videos or external knowledge.
- If the KB does not contain the answer, reply EXACTLY: "The answer is not present in the pdfs."
""")

    # --- Logic Based on Source ---
    from_video = from_kb = "The answer is not present in the documents."
    input_text_vid = PROMPT_VIDEO_ONLY.format(
        question=query,
        memory_context=memory_context,
        long_term=video_mem,
        video_context=video_context
    )
    raw_vid = llm.invoke(input_text_vid)
    from_video = getattr(raw_vid, "content", str(raw_vid)).strip()
    input_text_pdf = PROMPT_KB_ONLY.format(
        question=query,
        memory_context=memory_context,
        long_term=kb_mem,
        kb_context=kb_context
    )
    raw_pdf = llm.invoke(input_text_pdf)
    from_kb = getattr(raw_pdf, "content", str(raw_pdf)).strip()
    #  Video-only sources
    print("video context ",video_context)
    print("kb context ",kb_context)

    # --- Save Memory ---
    try:
        st.session_state.entity_memory.save_context(
            {"input": query},
            {"output": f"Video: {from_video}\nKB: {from_kb}"}
        )
    except Exception:
        pass
    print("from video",from_video)
    print("from_kb",from_kb)
    return {"video": from_video, "kb": from_kb}


def get_video_answer(query: str, video_index_tuple: Tuple[Any, Dict[str, Any]], llm, embeddings,
                     top_k=50, rerank_k=6, neighbor_window=2) -> Dict[str, Any]:
    print("inside the get_video_answer ")
    global video_QA_index
    out = {"snippet": "The answer is not present in the videos.", "timestamp": "", "source": ""}

    # Step 1: check long-term QA memory
    if video_QA_index:
        qa_docs = video_QA_index.similarity_search(query, k=50)
        if qa_docs and qa_docs[0].page_content:
            out["snippet"] = qa_docs[0].page_content
            out["timestamp"] = "N/A"
            out["source"] = "Video_QA_Memory"
            return out

    # Step 2: retrieve from vector index
    video_index, doc_store = video_index_tuple if isinstance(video_index_tuple, tuple) else (video_index_tuple, {})
    if not video_index:
        print("NO video index***********************************")
        return out

    docs = video_index.similarity_search(query, k=min(top_k, 20))
    if not docs:
        print("No docs found *****************************")
        return out

    # Step 3: rerank results
    try:
        if use_crossencoder:
            pairs = [[query, d.page_content] for d in docs]
            scores = RERANKER.predict(pairs)
            best_docs = [doc for doc, score in zip(docs, scores) if score > 0.3]
        else:
            best_docs = cosine_rerank(query, docs, embeddings, top_k=30)
    except Exception:
        best_docs = cosine_rerank(query, docs, embeddings, top_k=30)


    if not best_docs:
        print("no best docs found ********************")
        return out

    # Step 4: expand neighbors and merge chunks
    expanded = expand_with_neighbors(best_docs, doc_store, window=neighbor_window)
    merged_text = "\n".join([d.page_content for d in expanded])
    final_doc = Document(page_content=merged_text, metadata={"source_type": "video"})

    # Step 5: unified LLM generation (both video + KB keys returned)
    ans_dict = answer_from_docs(query, [final_doc], llm, source_type="video")
    snippet = ans_dict["video"]
    print("****Snippet*****",snippet)
    # Step 6: handle "no answer" case
    if snippet == "The answer is not present in the videos.":
        return out

    # Step 7: save QA to memory
    add_to_QA_index(query, snippet, "video", embeddings)

    # Step 8: collect source info (timestamps, etc.)
    source_info = {}
    for d in expanded:
        md = d.metadata or {}
        src = md.get("source", "unknown")
        if "end_time" in md:
            ts = f"{seconds_to_hhmmss(md['end_time'])}"
            source_info.setdefault(src, set()).add(ts)

    # Step 9: finalize output
    sources_summary = [f"{src} (timestamps: {', '.join(sorted(ts))})" for src, ts in source_info.items()]
    out["snippet"] = snippet
    out["timestamp"] = "; ".join(sources_summary) or "N/A"
    out["source"] = f"{len(source_info)} source(s)"
    return out

def get_kb_answer(query: str, pdf_index_tuple: Tuple[Any, Dict[str, Any]], llm, embeddings,
                  top_k=12, rerank_k=4, neighbor_window=2) -> Dict[str, Any]:
    global kb_QA_index
    out = {"answer": "The answer is not present in the pdfs.", "sources": []}
    # Step 1: check long-term QA memory
    if kb_QA_index:
        qa_docs = kb_QA_index.similarity_search(query, k=1)
        if qa_docs and qa_docs[0].page_content:
            out["answer"] = qa_docs[0].page_content
            out["sources"] = ["KB_QA_Memory"]
            return out
    # Step 2: fallback to original index
    # combined_index, doc_store = combined_index_tuple if isinstance(combined_index_tuple, tuple) else (combined_index_tuple, {})
    # if not combined_index:
    #     return out
    pdf_index, doc_store = pdf_index_tuple if isinstance(pdf_index_tuple, tuple) else (pdf_index_tuple, {})
    if not pdf_index:
        print("NO PDF index***********************************")
        return out
    docs = pdf_index.similarity_search(query, k=min(top_k, 30))
    if not docs:
        return out
    try:
        sel = crossencode_rerank(query, docs, top_k=rerank_k) if use_crossencoder else cosine_rerank(query, docs, embeddings, top_k=rerank_k)
    except Exception:
        sel = cosine_rerank(query, docs, embeddings, top_k=rerank_k)
    if not sel:
        return out
    expanded = expand_with_neighbors(sel, doc_store, window=neighbor_window)
    ans_dict = answer_from_docs(query, expanded, llm, source_type="pdf")
    ans = ans_dict["kb"]
    if ans == "The answer is not present in the pdfs.":
        return out
    # Step 3: save into QA memory
    add_to_QA_index(query, ans, "pdf", embeddings)
    # Collect sources
    pdf_sources, video_sources = [], []
    for d in sel:
        md = d.metadata or {}
        if md.get("source_type") == "video":
            v = md.get("source") or ""
            if v and v not in video_sources:
                video_sources.append(v)
        else:
            s = md.get("source") or ""
            if s and s not in pdf_sources:
                pdf_sources.append(s)
    out["answer"] = ans
    out["sources"] = pdf_sources + video_sources
    return out


# ===================== STREAMLIT UI (CHAT) =====================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "new_chat_count" not in st.session_state:
    st.session_state.new_chat_count = 0
if "pdf_files" not in st.session_state:
    st.session_state.pdf_files = []
if "video_jsons" not in st.session_state:
    st.session_state.video_jsons = []
if "video_urls_input" not in st.session_state:
    st.session_state.video_urls_input = ""
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""
if "current_chat_index" not in st.session_state:
    st.session_state.current_chat_index = None

st.set_page_config(page_title="📖 Knowledge Explore", layout="wide")
llm, embeddings = init_llm_and_embeddings()
# load_QA_indexes(embeddings)

# ✅ Use k=10 consistently for reliable memory
if "entity_memory" not in st.session_state:
    st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=10)

with st.sidebar:
    if st.button("➕ New Chat"):
        # Clear session for new chat
        st.session_state.messages = []
        st.session_state.pdf_files = []
        st.session_state.video_jsons = []
        st.session_state.video_urls_input = ""
        st.session_state.chat_input = ""
        st.session_state.new_chat_count += 1
        # ✅ Use k=10 here too
        st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=10)
        st.session_state.current_chat_index = None
        if callable(getattr(st, "rerun", None)):
            st.rerun()
        else:
            st.warning("Please refresh to start a new chat.")

    st.markdown("---")

    def generate_chat_title(messages):
        for msg in messages:
            if msg["role"] == "user" and msg["content"].strip():
                content = msg["content"].strip()
                return " ".join(content.split()[:6]) + ("..." if len(content.split()) > 6 else "")
        return "New Chat"

    st.markdown("### 🕘 Recent Chats")
    last_chats = st.session_state.chat_history[:5]
    if not last_chats:
        st.info("No previous chats.")
    else:
        for i, chat in enumerate(last_chats):
            chat_title = chat.get("title", f"Chat {i+1}")
            btn_key = f"load_chat_{i}"
            if st.button(chat_title, key=btn_key):
                st.session_state.messages = chat["messages"].copy()
                st.session_state.current_chat_index = i
                # ✅ DO NOT manually reconstruct memory — it causes corruption
                st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=10)
                st.rerun()

# --- Header ---
st.markdown("""
<div style="
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    background: linear-gradient(to right, #ffffff, #ffffff, #FFB84D);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
">
📖 Knowledge Explore
</div>
<div style="text-align: center; font-size: 20px; color: #FFD580; margin-top: 5px;">
Upload. Ask. Discover.
</div>
""", unsafe_allow_html=True)

# --- Load meta ---
meta = load_meta()
existing_pdfs = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith((".pdf", ".txt"))]
existing_vids = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith(".json")]
meta["pdfs"] = sorted(set([p for p in meta.get("pdfs", []) if os.path.exists(p)] + existing_pdfs))
meta["videos"] = sorted(set([v for v in meta.get("videos", []) if os.path.exists(v)] + existing_vids))
save_meta(meta)

# --- Load session vector stores ---
if "pdf_files" not in st.session_state:
    st.session_state.pdf_files = list(meta["pdfs"])
if "video_jsons" not in st.session_state:
    st.session_state.video_jsons = list(meta["videos"])

if "vs_pdf" not in st.session_state:
    st.session_state.vs_pdf, st.session_state.vs_pdf_store = load_index(INDEX_DIR_PDF, embeddings)
else:
    if isinstance(st.session_state.vs_pdf, tuple):
        st.session_state.vs_pdf, st.session_state.vs_pdf_store = st.session_state.vs_pdf
    else:
        st.session_state.vs_pdf_store = {}

if "vs_video" not in st.session_state:
    st.session_state.vs_video, st.session_state.vs_video_store = load_index(INDEX_DIR_VIDEO, embeddings)
else:
    if isinstance(st.session_state.vs_video, tuple):
        st.session_state.vs_video, st.session_state.vs_video_store = st.session_state.vs_video
    else:
        st.session_state.vs_video_store = {}

# --- File Upload ---
pdf_file = st.file_uploader(
    "Upload PDF/TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    key=f"uploaded_files_{st.session_state.new_chat_count}"
)

if pdf_file:
    for f in pdf_file:
        save_path = os.path.join(DATA_DIR, f.name)
        with open(save_path, "wb") as out:
            out.write(f.read())
        if save_path not in st.session_state.pdf_files:
            st.session_state.pdf_files.append(save_path)
        if save_path not in meta["pdfs"]:
            meta["pdfs"].append(save_path)
    save_meta(meta)

urls_str = st.text_input("Enter Video URLs (comma-separated)", key="video_urls_input")

# --- Process Button ---
if st.button("Process"):
    if urls_str.strip():
        urls = [u.strip() for u in urls_str.split(",") if u.strip()]
        with st.spinner("Adding new videos..."):
            add_count = 0
            for u in urls:
                jpath = transcribe_url(u)
                if jpath not in st.session_state.video_jsons:
                    st.session_state.video_jsons.append(jpath)
                    add_count += 1
                if jpath not in meta["videos"]:
                    meta["videos"].append(jpath)
            save_meta(meta)
        if add_count:
            st.success(f"Added {add_count} new video(s).")

    with st.spinner("Updating PDF index..."):
        vs_pdf, pdf_store, added_pdf_sources = update_index_incremental(
            st.session_state.pdf_files, embeddings, INDEX_DIR_PDF
        )
        st.session_state.vs_pdf = vs_pdf
        st.session_state.vs_pdf_store = pdf_store
        for f in st.session_state.pdf_files:
            src = _expected_source_for_file(f)
            if src in added_pdf_sources and f not in meta["indexed_pdfs"]:
                meta["indexed_pdfs"].append(f)
        save_meta(meta)

    with st.spinner("Updating Video index..."):
        vs_vid, vid_store, added_vid_sources = update_index_incremental(
            st.session_state.video_jsons, embeddings, INDEX_DIR_VIDEO
        )
        st.session_state.vs_video = vs_vid
        st.session_state.vs_video_store = vid_store
        for f in st.session_state.video_jsons:
            src = _expected_source_for_file(f)
            if src in added_vid_sources and f not in meta["indexed_videos"]:
                meta["indexed_videos"].append(f)
        save_meta(meta)

    # with st.spinner("Updating combined index..."):
    #     combined_files = st.session_state.pdf_files + st.session_state.video_jsons
    #     vs_comb, comb_store, added_comb_sources = update_index_incremental(
    #         combined_files, embeddings, INDEX_DIR_COMBINED
    #     )
    #     st.session_state.vs_combined = vs_comb
    #     st.session_state.vs_combined_store = comb_store
    #     for f in combined_files:
    #         src = _expected_source_for_file(f)
    #         if src in added_comb_sources and f not in meta["indexed_combined"]:
    #             meta["indexed_combined"].append(f)
    #     save_meta(meta)

    st.success("✅ Knowledge base updated. You can ask questions now.")
# --- Chat interface ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_q = st.chat_input("Ask your question…")

if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.spinner("Thinking…"):
        # Step 1: Get video-based answer
        video_part = get_video_answer(
            user_q,
            (st.session_state.get("vs_video"), st.session_state.get("vs_video_store", {})),
            llm,
            embeddings,
            top_k=12,
            rerank_k=10,
            neighbor_window=2,
        )
        print("videopart",video_part)

        # Step 2: Always call KB answer normally (no manual override needed)
        kb_part = get_kb_answer(
            user_q,
            (st.session_state.get("vs_pdf"), st.session_state.get("vs_pdf_store", {})),
            llm,
            embeddings,
            top_k=12,
            rerank_k=10,
            neighbor_window=1,
        )
        print("kbpart",kb_part)

    # Step 3: Display final combined result
    answer_md = f"""**From Video:** {video_part['snippet']}  
**Timestamp:** {video_part['timestamp'] or 'N/A'} • **Video Source:** {video_part['source'] or 'N/A'}

---

**From Knowledge Base (Summary):** {video_part['snippet']}

{kb_part['answer']}
"""

    with st.chat_message("assistant"):
        st.markdown(answer_md)

    st.session_state.messages.append({"role": "assistant", "content": answer_md})

    # --- Save/update chat history ---
    title = generate_chat_title(st.session_state.messages)

    if st.session_state.current_chat_index is not None:
        chat_to_move = st.session_state.chat_history.pop(st.session_state.current_chat_index)
        chat_to_move["messages"] = st.session_state.messages.copy()
        chat_to_move["title"] = title
        st.session_state.chat_history.insert(0, chat_to_move)
        st.session_state.current_chat_index = 0
    else:
        st.session_state.chat_history.insert(0, {
            "title": title,
            "messages": st.session_state.messages.copy()
        })
        st.session_state.current_chat_index = 0

    st.session_state.chat_history = st.session_state.chat_history[:5]

st.caption("💡 Tip: Click Process only when adding new files/videos — old knowledge is reused.")