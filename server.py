"""
server.py  —  FastAPI replacement for app.py (Streamlit removed)
================================================================
All logic modules are untouched.  Only the UI layer changes.

INSTALL (once):
    pip install fastapi uvicorn python-multipart

RUN:
    uvicorn server:app --reload --port 8000

Then open:  http://localhost:8000
"""

import os, sys, uuid, warnings
from pathlib import Path
from typing  import Optional, List

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", message=".*migrating_memory.*")

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── FastAPI ────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Your existing modules (NOT modified at all) ────────────────────────────────
try:
    from langchain_classic.memory import ConversationEntityMemory
except ImportError:
    from langchain.memory import ConversationEntityMemory

from modules.config import (
    DATA_DIR, INDEX_DIR_PDF, INDEX_DIR_VIDEO,
    TOP_K_RETRIEVAL, RERANK_K, NEIGHBOUR_WINDOW, get_logger,
)
from modules.llm_client        import init_llm_and_embeddings
from modules.meta_store        import load_meta, save_meta, scan_data_dir
from modules.transcription     import transcribe, _get_ffmpeg_dir_for_ytdlp
from modules.vector_store      import load_index, update_index_incremental
from modules.document_loader   import expected_source_for_file
from modules.query_handler     import get_combined_answer
# ──────────────────────────────────────────────────────────────────────────────

logger = get_logger("server")

app = FastAPI(title="MediaMind")

# serve the UI
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse((Path(__file__).parent / "static" / "index.html").read_text())


# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP  — mirrors what _startup_load() and _init_session() did in app.py
# ══════════════════════════════════════════════════════════════════════════════

logger.info("Initialising LLM and embeddings…")
llm, embeddings = init_llm_and_embeddings()

# ffmpeg check (same as @st.cache_resource _preload_ffmpeg)
_ffmpeg_dir = _get_ffmpeg_dir_for_ytdlp()
if _ffmpeg_dir:
    logger.info("Bundled ffmpeg ready: %s", _ffmpeg_dir)
else:
    logger.warning("Bundled ffmpeg not found — URL transcription may fail")

# load indexes on startup (same as _startup_load)
logger.info("Loading indexes from disk…")
meta          = load_meta()
meta          = scan_data_dir(meta)
save_meta(meta)

vs_pdf,   pdf_store = load_index(INDEX_DIR_PDF,   embeddings)
vs_video, vid_store = load_index(INDEX_DIR_VIDEO, embeddings)

logger.info(
    "Startup complete — PDF index: %s | Video index: %s",
    "loaded" if vs_pdf   else "empty",
    "loaded" if vs_video else "empty",
)

# In-memory state that used to live in st.session_state.
# Keyed by session_id so multiple browser tabs work independently.
# Each value mirrors the fields in _init_session().
_sessions: dict[str, dict] = {}

def _new_session(session_id: str) -> dict:
    return {
        "session_id":    session_id,
        "messages":      [],               # [{role, content}]
        "pdf_files":     list(meta.get("pdfs",   [])),
        "video_jsons":   list(meta.get("videos", [])),
        "raw_video_files": [],
        "vs_pdf":        vs_pdf,
        "vs_pdf_store":  pdf_store,
        "vs_video":      vs_video,
        "vs_video_store":vid_store,
        "entity_memory": ConversationEntityMemory(llm=llm, k=10),
    }

def _get_or_create(session_id: str) -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = _new_session(session_id)
    return _sessions[session_id]


# ══════════════════════════════════════════════════════════════════════════════
#  REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════════════

class NewSessionResp(BaseModel):
    session_id: str
    has_index:  bool        # whether any index already exists

class ProcessURLReq(BaseModel):
    session_id: str
    url:        str

class ProcessStatusResp(BaseModel):
    session_id:  str
    indexed_pdfs:   int
    indexed_videos: int
    message:     str

class AskReq(BaseModel):
    session_id: str
    question:   str

class SourceChunk(BaseModel):
    ref:   str      # timestamp or filename
    text:  str      # short snippet
    score: float    # 0–1  (we derive this from has_video / has_kb)

class AskResp(BaseModel):
    answer:      str
    has_video:   bool
    has_kb:      bool
    sources:     List[SourceChunk]


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTE 1 — Create / open a session
#  Frontend calls this on "New Session" click
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/session/new", response_model=NewSessionResp)
async def new_session():
    sid = str(uuid.uuid4())
    _sessions[sid] = _new_session(sid)
    logger.info("New session created: %s", sid)
    return NewSessionResp(
        session_id=sid,
        has_index=(vs_pdf is not None or vs_video is not None),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTE 2a — Upload a PDF / TXT / video file
#  Mirrors the st.file_uploader blocks in app.py
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/upload/file")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    sess = _get_or_create(session_id)

    save_path = os.path.join(DATA_DIR, file.filename)
    with open(save_path, "wb") as fh:
        fh.write(await file.read())

    ext = Path(file.filename).suffix.lower()
    video_exts = {".mp4",".mkv",".avi",".mov",".webm",".m4v",".mp3",".wav",".aac",".flac",".m4a"}

    m = load_meta()
    if ext in video_exts:
        sess["raw_video_files"].append(save_path)
        logger.info("Video file uploaded to session %s: %s", session_id, file.filename)
    else:
        if save_path not in sess["pdf_files"]:
            sess["pdf_files"].append(save_path)
        if save_path not in m["pdfs"]:
            m["pdfs"].append(save_path)
        save_meta(m)
        logger.info("PDF/TXT uploaded to session %s: %s", session_id, file.filename)

    return {"filename": file.filename, "type": "video" if ext in video_exts else "pdf"}


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTE 2b — Submit a video URL
#  Mirrors the video_urls_input text box
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/upload/url")
async def upload_url(req: ProcessURLReq):
    sess = _get_or_create(req.session_id)

    # We don't transcribe yet — just store the URL.
    # Transcription happens when /api/process is called.
    sess.setdefault("pending_urls", [])
    if req.url not in sess["pending_urls"]:
        sess["pending_urls"].append(req.url)

    logger.info("URL queued for session %s: %s", req.session_id, req.url)
    return {"queued": req.url}


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTE 3 — Process  (the big "Process" button in Streamlit)
#  Transcribes URLs + video files, builds / updates both indexes.
#  This is the slow call — frontend shows the progress bar while waiting.
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/process", response_model=ProcessStatusResp)
async def process(session_id: str):
    sess  = _get_or_create(session_id)
    m     = load_meta()
    msgs  = []

    # ── Transcribe pending URLs (mirrors "Transcribe urls" block) ──
    for url in sess.get("pending_urls", []):
        try:
            jpath = transcribe(url)                     # ← your function, unchanged
            if jpath not in sess["video_jsons"]:
                sess["video_jsons"].append(jpath)
            if jpath not in m["videos"]:
                m["videos"].append(jpath)
            logger.info("URL transcribed: %s", url)
        except Exception as exc:
            logger.error("Transcription failed for %s: %s", url, exc)
            raise HTTPException(400, detail=f"Could not transcribe {url}: {exc}")

    sess["pending_urls"] = []
    save_meta(m)

    # ── Transcribe uploaded video files ──
    for vpath in sess.get("raw_video_files", []):
        try:
            jpath = transcribe(vpath)                   # ← your function, unchanged
            if jpath not in sess["video_jsons"]:
                sess["video_jsons"].append(jpath)
            if jpath not in m["videos"]:
                m["videos"].append(jpath)
        except Exception as exc:
            logger.error("Video transcription failed %s: %s", vpath, exc)
            raise HTTPException(400, detail=f"Transcription failed for {os.path.basename(vpath)}: {exc}")

    sess["raw_video_files"] = []
    save_meta(m)

    # ── Update PDF index (mirrors "Update PDF index" block) ──
    try:
        vs_p, p_store, added_pdf = update_index_incremental(   # ← your function
            sess["pdf_files"], embeddings, INDEX_DIR_PDF
        )
        sess["vs_pdf"]       = vs_p
        sess["vs_pdf_store"] = p_store

        for f in sess["pdf_files"]:
            src = expected_source_for_file(f)
            if src in added_pdf and f not in m["indexed_pdfs"]:
                m["indexed_pdfs"].append(f)
        save_meta(m)
        msgs.append(f"Indexed {len(added_pdf)} new document(s)." if added_pdf else "No new documents.")
    except Exception as exc:
        logger.error("PDF indexing failed: %s", exc)
        raise HTTPException(500, detail=f"Document indexing failed: {exc}")

    # ── Update Video index ──
    try:
        vs_v, v_store, added_vid = update_index_incremental(   # ← your function
            sess["video_jsons"], embeddings, INDEX_DIR_VIDEO
        )
        sess["vs_video"]       = vs_v
        sess["vs_video_store"] = v_store

        for f in sess["video_jsons"]:
            src = expected_source_for_file(f)
            if src in added_vid and f not in m["indexed_videos"]:
                m["indexed_videos"].append(f)
        save_meta(m)
        msgs.append(f"Indexed {len(added_vid)} new video(s)." if added_vid else "No new videos.")
    except Exception as exc:
        logger.error("Video indexing failed: %s", exc)
        raise HTTPException(500, detail=f"Video indexing failed: {exc}")

    return ProcessStatusResp(
        session_id=session_id,
        indexed_pdfs=len(m.get("indexed_pdfs", [])),
        indexed_videos=len(m.get("indexed_videos", [])),
        message=" | ".join(msgs),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTE 4 — Ask a question
#  Mirrors the `if user_q:` block in app.py — calls get_combined_answer()
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/ask", response_model=AskResp)
async def ask(req: AskReq):
    sess = _get_or_create(req.session_id)
    logger.info("Query [%s]: %.120s", req.session_id, req.question)

    try:
        result = get_combined_answer(              # ← your function, unchanged
            req.question,
            (sess["vs_video"], sess["vs_video_store"]),
            (sess["vs_pdf"],   sess["vs_pdf_store"]),
            llm,
            embeddings,
            sess["entity_memory"],
            top_k=TOP_K_RETRIEVAL,
            rerank_k=RERANK_K,
            neighbour_window=NEIGHBOUR_WINDOW,
        )
    except Exception as exc:
        logger.error("get_combined_answer failed: %s", exc)
        raise HTTPException(500, detail=str(exc))

    # ── Build the formatted answer (same logic as app.py "Format response") ──
    sections = []
    sources:  List[SourceChunk] = []

    if result["has_video"]:
        sections.append(
            f"**From Video**\n\n{result['video_answer']}\n\n"
            f"> **Timestamp**: {result['video_timestamp']}  \n"
            f"> **Source**: {result['video_source']}"
        )
        sources.append(SourceChunk(
            ref=result["video_timestamp"],
            text=result["video_source"],
            score=0.9,
        ))
    else:
        sections.append("**From Video**\n\n_No relevant content found in uploaded videos._")

    sections.append("---")

    if result["has_kb"]:
        kb_src = ", ".join(result["kb_sources"]) if result["kb_sources"] else "N/A"
        sections.append(
            f"**From Knowledge Base**\n\n{result['kb_answer']}\n\n"
            f"> **Sources**: {kb_src}"
        )
        for src in (result.get("kb_sources") or []):
            sources.append(SourceChunk(ref=src, text="", score=0.8))
    else:
        sections.append("**From Knowledge Base**\n\n_No relevant content found in uploaded documents._")

    answer = "\n\n".join(sections)

    # save to session message history
    sess["messages"].append({"role": "user",      "content": req.question})
    sess["messages"].append({"role": "assistant", "content": answer})

    logger.info("Answer generated — has_video=%s, has_kb=%s", result["has_video"], result["has_kb"])

    return AskResp(
        answer=answer,
        has_video=result["has_video"],
        has_kb=result["has_kb"],
        sources=sources,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTE 5 — Session history  (sidebar "Recent chats")
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/sessions")
async def list_sessions():
    out = []
    for sid, s in _sessions.items():
        msgs = s["messages"]
        # generate title same way as _generate_chat_title in app.py
        title = "New Session"
        for m in msgs:
            if m["role"] == "user" and m["content"].strip():
                words = m["content"].split()
                title = " ".join(words[:6]) + ("…" if len(words) > 6 else "")
                break
        out.append({
            "session_id":  sid,
            "title":       title,
            "msg_count":   len([m for m in msgs if m["role"] == "user"]),
        })
    return out


@app.get("/api/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    sess = _sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    return sess["messages"]
