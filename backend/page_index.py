import os
import re
import json
import logging
import requests
from typing import List, Dict, Tuple, Optional

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from google import genai

# Import rag module for token tracking
import rag

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# CLIENTS / MODELS


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

gemini_client = genai.Client(api_key=GOOGLE_API_KEY)


model = SentenceTransformer("BAAI/bge-small-en-v1.5")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "documents")
TREE_INDEX_PATH = os.path.join(BASE_DIR, "tree_index.json")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "tree_embeddings.npy")

# In-memory cache so we never rebuild / reload from disk mid-session.
_TREE_INDEX: Optional[List[Dict]] = None
_TREE_EMBEDDINGS: Optional[np.ndarray] = None



# LLM CALL WITH GEMINI -> GROQ FALLBACK
# NOW WITH TOKEN TRACKING


def call_llm(prompt: str, max_tokens: int = 1024) -> str:
    """Call Gemini first; on any failure (quota, network, etc.) fall back to Groq.
    
    Captures real usage from both Gemini and Groq responses and reports via
    rag._record_usage() so it folds into the request's running total.
    """
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        text = (response.text or "").strip()
        
        # Capture Gemini usage
        usage = getattr(response, "usage_metadata", None)
        if usage:
            rag._record_usage(
                prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            )
            log.info(f"[PAGE_INDEX] Gemini usage recorded | prompt={getattr(usage, 'prompt_token_count', 0)} | completion={getattr(usage, 'candidates_token_count', 0)}")
        
        if text:
            return text
        raise ValueError("Empty response from Gemini")
    except Exception as e:
        log.warning(f"[PAGE_INDEX] Gemini call failed ({e}) — trying Groq fallback...")

    if not GROQ_API_KEY:
        log.error("[PAGE_INDEX] No GROQ_API_KEY set, cannot fall back.")
        raise RuntimeError("Both Gemini and Groq are unavailable.")

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        
        # Capture Groq usage from HTTP response
        usage = data.get("usage", {})
        if usage:
            rag._record_usage(
                prompt_tokens=usage.get("prompt_tokens", 0) or 0,
                completion_tokens=usage.get("completion_tokens", 0) or 0,
            )
            log.info(f"[PAGE_INDEX] Groq usage recorded | prompt={usage.get('prompt_tokens', 0)} | completion={usage.get('completion_tokens', 0)}")
        
        log.info("[PAGE_INDEX] Groq fallback succeeded")
        return text
    except Exception as e:
        log.error(f"[PAGE_INDEX] Groq fallback also failed: {e}")
        raise



# STEP 1: BUILD / CACHE TREE INDEX FROM DOCUMENTS


def parse_headings(text: str, page_num: int, file: str) -> List[Dict]:
    """
    Best-effort heading detection for nicer source titles.
    Real PDF text rarely contains literal '#' markdown, so this is a soft
    heuristic (short, title-cased or ALL-CAPS lines near the top of a page)
    and is allowed to return nothing — the page-level node is the guaranteed
    fallback, not this.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    sections = []

    for idx, line in enumerate(lines[:15]):  # only scan near the top of the page
        is_short = 3 <= len(line) <= 80
        is_capsy = line.isupper() or (line.istitle() and len(line.split()) <= 10)
        if is_short and is_capsy and not line.endswith((".", ",", ";")):
            sections.append({
                "id": f"page_{page_num}_h_{len(sections)}",
                "level": 1,
                "title": line,
                "children": []
            })

    return sections


def build_tree_index() -> List[Dict]:
    """Read PDFs once and create a page-level tree structure (with optional
    detected headings for nicer titles). Every page becomes a node — this is
    what guarantees sources/nodes are never empty."""
    log.info("[TREE_INDEX] Building tree structure from PDFs (this happens once)...")

    tree_data = []

    if not os.path.isdir(PDF_FOLDER):
        log.error(f"[TREE_INDEX] PDF folder not found: {PDF_FOLDER}")
        return tree_data

    for file in os.listdir(PDF_FOLDER):
        if not file.endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_FOLDER, file)
        try:
            log.info(f"[TREE_INDEX] Loading {file}...")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            for page_num, doc in enumerate(docs):
                page_content = doc.page_content
                if not page_content or not page_content.strip():
                    continue

                sections = parse_headings(page_content, page_num, file)

                # Always append the page — sections may be empty, that's fine.
                tree_data.append({
                    "file": file,
                    "page": page_num,
                    "content": page_content,
                    "sections": sections
                })

        except Exception as e:
            log.error(f"[TREE_INDEX] Failed to load {file}: {e}")
            continue

    with open(TREE_INDEX_PATH, "w") as f:
        json.dump(tree_data, f, indent=2)

    log.info(f"[TREE_INDEX] Tree index built: {len(tree_data)} pages")
    return tree_data


def _load_tree_from_disk_or_build() -> List[Dict]:
    if os.path.exists(TREE_INDEX_PATH):
        try:
            with open(TREE_INDEX_PATH, "r") as f:
                data = json.load(f)
            if data:
                log.info(f"[TREE_INDEX] Loaded {len(data)} pages from disk cache")
                return data
        except Exception as e:
            log.warning(f"[TREE_INDEX] Failed to read cache, rebuilding: {e}")
    return build_tree_index()


def get_tree_index() -> Tuple[List[Dict], np.ndarray]:
    
    """Returns (tree_index, page_embeddings), built once and cached both in
    memory (for the rest of this process) and on disk (so the NEXT process
    restart doesn't have to re-embed 1000+ pages from scratch, which is what
    made the first Page Index query after a restart take several minutes)."""
    global _TREE_INDEX, _TREE_EMBEDDINGS

    if _TREE_INDEX is not None and _TREE_EMBEDDINGS is not None:
        return _TREE_INDEX, _TREE_EMBEDDINGS

    tree_data = _load_tree_from_disk_or_build()

    if not tree_data:
        _TREE_INDEX = []
        _TREE_EMBEDDINGS = np.zeros((0, 384), dtype=np.float32)
        return _TREE_INDEX, _TREE_EMBEDDINGS

    # Try loading cached embeddings from disk first. Only valid if the count
    # matches the current tree_index — otherwise the PDFs changed since the
    # cache was written and we must re-embed to stay in sync.
    if os.path.exists(EMBEDDINGS_PATH):
        try:
            cached_embeddings = np.load(EMBEDDINGS_PATH)
            if cached_embeddings.shape[0] == len(tree_data):
                log.info(f"[TREE_INDEX] Loaded {cached_embeddings.shape[0]} cached embeddings from disk")
                _TREE_INDEX = tree_data
                _TREE_EMBEDDINGS = cached_embeddings.astype(np.float32)
                return _TREE_INDEX, _TREE_EMBEDDINGS
            else:
                log.warning(
                    f"[TREE_INDEX] Cached embeddings count ({cached_embeddings.shape[0]}) "
                    f"doesn't match tree_index ({len(tree_data)}) — recomputing."
                )
        except Exception as e:
            log.warning(f"[TREE_INDEX] Failed to load cached embeddings, recomputing: {e}")

    log.info(f"[TREE_INDEX] Embedding {len(tree_data)} pages (one-time cost)...")
    texts = [page["content"][:2000] for page in tree_data]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    _TREE_INDEX = tree_data
    _TREE_EMBEDDINGS = np.asarray(embeddings, dtype=np.float32)
    log.info("[TREE_INDEX] Embeddings cached in memory")

    try:
        np.save(EMBEDDINGS_PATH, _TREE_EMBEDDINGS)
        log.info(f"[TREE_INDEX] Embeddings saved to disk: {EMBEDDINGS_PATH}")
    except Exception as e:
        log.warning(f"[TREE_INDEX] Failed to save embeddings to disk: {e}")

    return _TREE_INDEX, _TREE_EMBEDDINGS



# STEP 2: EMBEDDING-BASED NODE SELECTION (replaces the LLM ID-picking loop)


def retrieve_top_pages(query: str, tree_index: List[Dict], embeddings: np.ndarray,
                        top_k: int = 5) -> List[Tuple[Dict, float]]:
    if len(tree_index) == 0:
        return []

    query_emb = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
    sims = embeddings @ query_emb  # cosine similarity, both sides normalized
    top_k = min(top_k, len(tree_index))
    top_idx = np.argsort(-sims)[:top_k]

    return [(tree_index[i], float(sims[i])) for i in top_idx]



# STEP 3: ANSWER GENERATION


def generate_answer_from_content(query: str, content: List[str]) -> str:
    combined_content = "\n\n".join(content)[:4000]

    prompt = f"""
You are an expert medical assistant.

Using ONLY the provided content, answer this question completely:
{query}

Instructions:

- Answer in full sentences
- Be comprehensive
- Include all relevant types, precautions, symptoms, treatments
- Do not copy verbatim, synthesize naturally
- Never mention source documents

Content:
{combined_content}

Answer:
"""
    try:
        return call_llm(prompt, max_tokens=800)
    except Exception as e:
        log.error(f"[PAGE_INDEX] Answer generation failed: {e}")
        return "Could not generate an answer right now — please try again shortly."



# STEP 4: BUILD HIERARCHICAL SOURCES FOR THE FRONTEND
def build_sources_hierarchy(top_pages: List[Tuple[Dict, float]]) -> Dict:
    hierarchy = {
        "title": "Page Index Sources",
        "node_id": "root",
        "nodes": []
    }

    for page, score in top_pages:
        page_node = {
            "title": f"{page['file']} - Page {page['page'] + 1}",
            "node_id": f"page_{page['page']}_{page['file']}",
            "page": page["page"],
            "file": page["file"],
            "score": round(score, 3),
            "nodes": []
        }

        for section in page.get("sections", []):
            page_node["nodes"].append({
                "title": section["title"],
                "node_id": section["id"],
                "level": 1,
                "nodes": []
            })

        hierarchy["nodes"].append(page_node)

    return hierarchy



# MAIN PIPELINE


def page_index_pipeline(query: str, top_k: int = 5) -> Dict:
    """
    1. Load cached tree index + embeddings (built once per process)
    2. Retrieve top-k relevant pages via semantic similarity (no LLM call)
    3. Generate one answer from that content (single LLM call, Groq fallback)
    4. Return answer + hierarchical sources
    
    All LLM usage is now tracked via rag._record_usage() and will be folded
    into the request's running total when main.py calls rag.get_tracked_usage().
    """
    log.info(f"[PAGE_INDEX] Starting pipeline for query: {query}")

    tree_index, embeddings = get_tree_index()

    if not tree_index:
        log.warning("[PAGE_INDEX] No tree index available")
        return {
            "query": query,
            "answer": "No documents available in the knowledge base.",
            "sources": {},
            "success": False
        }

    top_pages = retrieve_top_pages(query, tree_index, embeddings, top_k=top_k)

    # If even the best match is a poor semantic fit, say so rather than
    # forcing an answer out of irrelevant content.
    best_score = top_pages[0][1] if top_pages else 0.0
    if best_score < 0.2:
        log.info(f"[PAGE_INDEX] Best match score too low ({best_score:.3f})")
        return {
            "query": query,
            "answer": "No information found in the knowledge base for this query.",
            "sources": build_sources_hierarchy(top_pages),
            "success": True
        }

    content = [page["content"] for page, _ in top_pages]
    answer = generate_answer_from_content(query, content)
    sources = build_sources_hierarchy(top_pages)

    log.info(f"[PAGE_INDEX] Done | pages_used={len(top_pages)} | best_score={best_score:.3f}")

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "success": True
    }


# STREAMING WRAPPER (for the /page-index-stream SSE endpoint)


def page_index_stream(query: str, top_k: int = 5, chunk_size: int = 40):
    """
    Generator yielding SSE-formatted strings compatible with the frontend's
    parser (expects 'data: {json}\\n' lines with type 'text' and 'done').
    The pipeline itself is not token-streamed by Gemini/Groq here, so we
    compute the full answer once, then stream it out in small chunks for a
    smooth typing effect in the UI.
    """
    try:
        result = page_index_pipeline(query, top_k=top_k)
    except Exception as e:
        log.error(f"[PAGE_INDEX] Pipeline failed: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        return

    answer = result.get("answer", "")

    for i in range(0, len(answer), chunk_size):
        piece = answer[i:i + chunk_size]
        yield f"data: {json.dumps({'type': 'text', 'content': piece})}\n\n"

    done_payload = {
        "type": "done",
        "page_index_sources": result.get("sources", {}),
        "success": result.get("success", False),
    }
    yield f"data: {json.dumps(done_payload)}\n\n"


if __name__ == "__main__":
    # Quick manual test
    q = "heart failure and why does this happen"
    r = page_index_pipeline(q)
    print(json.dumps(r, indent=2)[:2000])
