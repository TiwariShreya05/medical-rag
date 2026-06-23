import os
import logging
import base64

print("NEW RAG FILE LOADED")

# LOGGING SETUP
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer, CrossEncoder

from rank_bm25 import BM25Okapi

load_dotenv()

from google import genai
from google.genai import types

api_key = os.getenv("GOOGLE_API_KEY")

log.info(f"API KEY FOUND: {api_key[:10] if api_key else 'NONE'}")

gemini_client = genai.Client(api_key=api_key)


def generate_with_retry(model, contents, retries=3, delay=10):
    """Calls Gemini and retries on transient errors (like 503 overload) before giving up.
    Returns the response text, or "" if every attempt fails."""
    import time
    for attempt in range(retries + 1):
        try:
            response = gemini_client.models.generate_content(model=model, contents=contents)
            return response.text.strip() if response.text else ""
        except Exception as e:
            log.warning(f"Gemini call failed (attempt {attempt + 1}/{retries + 1}) | model={model} | error={e}")
            if attempt < retries:
                time.sleep(delay)
    return ""


# EMBEDDING MODEL
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# HuggingFace embeddings for SemanticChunker
hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# QDRANT
client = None
COLLECTION_NAME = "medical_docs"

bm25 = None
bm25_chunks = []

# absolute paths so cache always saves/loads from same place as rag.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BM25_CACHE_PATH = os.path.join(BASE_DIR, "bm25_cache.pkl")
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db_v2")
PDF_FOLDER = os.path.join(BASE_DIR, "documents")

log.info(f"BASE_DIR: {BASE_DIR}")
log.info(f"BM25_CACHE_PATH: {BM25_CACHE_PATH}")


def get_web_grounded_answer(query: str) -> str:
    """
    Fallback for when the local PDF knowledge base has nothing good for this
    query. Uses Gemini's built-in Google Search tool so the model grounds its
    answer in live web results instead of just guessing from training data.
    """
    prompt = f"""You are an expert medical and biology assistant. Answer the
following question clearly and accurately using current, reliable
information. Use simple language. If this is a medical question, add a brief
note to consult a doctor for personal medical advice.

Question: {query}

Answer:"""
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            ),
        )
        return response.text.strip() if response.text else ""
    except Exception as e:
        log.warning(f"Web-grounded answer failed: {e}")
        return ""


def get_client():
    global client
    if client is None:
        client = QdrantClient(path=QDRANT_PATH)
    return client


def semantic_recursive_split(all_docs):
    log.info("Stage 1: Semantic chunking...")
    semantic_splitter = SemanticChunker(hf_embeddings)

    semantic_docs = []
    for doc in all_docs:
        semantic_docs.extend(
            semantic_splitter.create_documents([doc.page_content])
        )
    log.info(f"Semantic chunks: {len(semantic_docs)}")

    log.info("Stage 2: Recursive chunking...")
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    final_chunks = []
    for doc in semantic_docs:
        splits = recursive_splitter.create_documents([doc.page_content])
        final_chunks.extend(splits)

    log.info(f"Final chunks after recursive split: {len(final_chunks)}")
    return final_chunks


def build_vector_db():
    all_docs = []

    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_FOLDER, file))
            docs = loader.load()
            all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    final_chunks = splitter.split_documents(all_docs)

    global bm25, bm25_chunks
    bm25_chunks = final_chunks
    tokenized_corpus = [chunk.page_content.lower().split() for chunk in final_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    log.info(f"Total chunks created: {len(final_chunks)}")
    return len(final_chunks)


def create_qdrant_collection():
    client = get_client()
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    return "Collection Created"


def generate_embeddings():
    embedding = model.encode("What is diabetes?")
    return len(embedding)


def ingest_documents():
    client = get_client()
    all_docs = []

    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, file)
            try:
                log.info(f"Loading PDF: {file}")
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                all_docs.extend(docs)
                log.info(f"SUCCESS: {file} | Pages={len(docs)}")
            except Exception as e:
                log.error(f"FAILED PDF: {file}")
                log.error(str(e))
                continue

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    final_chunks = splitter.split_documents(all_docs)

    global bm25, bm25_chunks
    bm25_chunks = final_chunks
    tokenized_corpus = [chunk.page_content.lower().split() for chunk in final_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    texts = [chunk.page_content for chunk in final_chunks]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    points = []
    for idx, (chunk, embedding) in enumerate(zip(final_chunks, embeddings)):
        points.append(PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload={
                "text": chunk.page_content,
                "page": chunk.metadata.get("page", "Unknown"),
                "source": chunk.metadata.get("source", "Unknown"),
                "chunk_id": idx
            }
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    log.info(f"Stored {len(points)} vectors")

    import pickle
    with open(BM25_CACHE_PATH, "wb") as f:
        pickle.dump(bm25_chunks, f)
    log.info(f"BM25 cache saved: {BM25_CACHE_PATH}")

    return len(points)


def refine_query(query, context_text=""):
    prompt = f"""
You are a medical search assistant.

Improve the query to retrieve better medical documents.

Rules:
- Keep it short
- Add medical synonyms
- Do NOT answer the question

Query: {query}

Context (optional):
{context_text}
"""
    refined = generate_with_retry("gemini-2.5-flash-lite", prompt)
    if refined:
        log.info(f"Refined query: {refined}")
        return refined
    return query


def bm25_search(query, top_k=10):
    global bm25, bm25_chunks
    if bm25 is None:
        return []

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in ranked_indices:
        chunk = bm25_chunks[idx]
        results.append({
            "chunk_id": idx,
            "page": chunk.metadata.get("page"),
            "source_file": chunk.metadata.get("source"),
            "score": float(scores[idx]),
            "score_type": "bm25",
            "text": chunk.page_content
        })
    return results


def expand_query(query):
    prompt = f"""
Expand this medical query with synonyms.

Return ONLY the improved query.

Query:
{query}
"""
    expanded = generate_with_retry("gemini-2.5-flash-lite", prompt)
    return expanded if expanded else query


def search_similar_chunks(query):
    client = get_client()
    expanded_query = expand_query(query)

    log.info(f"Original query: {query}")
    log.info(f"Expanded query: {expanded_query}")

    query_embedding = model.encode(expanded_query, normalize_embeddings=True).tolist()

    vector_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=15,
        with_payload=True
    ).points

    retrieved_chunks = []
    for hit in vector_results:
        if hit.payload is None:
            continue
        retrieved_chunks.append({
            "chunk_id": hit.payload.get("chunk_id"),
            "page": hit.payload.get("page", "Unknown"),
            "source_file": hit.payload.get("source", "Unknown"),
            "score": float(hit.score),
            "score_type": "vector",
            "text": hit.payload.get("text", "")
        })

    retrieved_chunks = [c for c in retrieved_chunks if c["score"] > 0.4]
    retrieved_chunks = sorted(retrieved_chunks, key=lambda x: x["score"], reverse=True)[:10]

    bm25_results = bm25_search(expanded_query, top_k=15)
    combined_chunks = retrieved_chunks + bm25_results

    unique_chunks = {}
    for chunk in combined_chunks:
        key = (chunk.get("text", ""), chunk.get("page"), chunk.get("source_file"))
        unique_chunks[key] = chunk
    combined_chunks = list(unique_chunks.values())

    log.info(f"Vector results: {len(retrieved_chunks)}")
    log.info(f"BM25 results: {len(bm25_results)}")
    log.info(f"Combined after dedup: {len(combined_chunks)}")

    pairs = [(query, c["text"]) for c in combined_chunks if c.get("text")]
    scores = reranker.predict(pairs)

    for chunk, score in zip(combined_chunks, scores):
        chunk["rerank_score"] = float(score)

    combined_chunks.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

    filtered = [c for c in combined_chunks if c.get("rerank_score", 0) > 1.5]
    if len(filtered) == 0:
        filtered = combined_chunks[:3]

    return filtered[:3]


def recursive_search(query, max_retries=1, threshold=0.35):
    current_query = query
    best_results = []

    for attempt in range(max_retries + 1):
        log.info(f"--- RETRIEVAL ATTEMPT {attempt + 1} ---")
        log.info(f"Query: {current_query}")

        results = search_similar_chunks(current_query)
        if not results:
            continue

        best_score = max((r.get("rerank_score", 0) for r in results), default=0)
        log.info(f"Best rerank score: {best_score:.4f}")

        if best_score > max((r.get("rerank_score", 0) for r in best_results), default=0):
            best_results = results

        if best_score >= threshold:
            log.info("Good results found. Stopping recursion.")
            break

        context_text = "\n".join([r["text"] for r in results[:2]])
        current_query = refine_query(current_query, context_text)

    return best_results


def get_llm_answer(query, context):
    context_text = "\n\n".join(c["text"][:1200] for c in context[:3])

    prompt = f"""
You are an expert medical and biology assistant.

Use the provided context to answer the user's question.

Instructions:
- First answer the question directly.
- Synthesize information from all relevant passages.
- Do NOT copy text from the context.
- Do NOT quote large chunks.
- Ignore unrelated information.
- Explain concepts clearly and completely.
- Use bullet points when useful.
- If the context contains the answer, provide a complete educational response.
- If the answer is not fully available in the context, say what is known and mention the limitation.
- Never mention retrieved chunks or document passages.

Context:
{context_text}

Question:
{query}

Answer:
"""

    answer = generate_with_retry("gemini-2.5-flash", prompt)
    if answer:
        return answer
    return (
        "I found relevant information in the documents, "
        "but the language model is currently unavailable. "
        "Please check the retrieved chunks below for the supporting evidence."
    )


def _rebuild_bm25_only():
    global bm25, bm25_chunks
    import pickle

    log.info(f"Checking BM25 cache at: {BM25_CACHE_PATH}")

    if os.path.exists(BM25_CACHE_PATH):
        log.info("Loading BM25 from cache...")
        with open(BM25_CACHE_PATH, "rb") as f:
            bm25_chunks = pickle.load(f)
        tokenized_corpus = [chunk.page_content.lower().split() for chunk in bm25_chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        log.info(f"BM25 loaded from cache: {len(bm25_chunks)} chunks")
        return

    log.info("No cache found. Building BM25 from PDF...")
    all_docs = []

    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, file)
            try:
                log.info(f"Loading PDF: {file}")
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                all_docs.extend(docs)
                log.info(f"SUCCESS: {file} | Pages={len(docs)}")
            except Exception as e:
                log.error(f"FAILED PDF: {file}")
                log.error(str(e))
                continue

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    bm25_chunks = splitter.split_documents(all_docs)
    log.info(f"BM25 chunks built: {len(bm25_chunks)}")

    tokenized_corpus = [chunk.page_content.lower().split() for chunk in bm25_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_CACHE_PATH, "wb") as f:
        pickle.dump(bm25_chunks, f)
    log.info(f"BM25 built and cached at: {BM25_CACHE_PATH} | chunks: {len(bm25_chunks)}")


def initialize():
    global client
    client = get_client()

    try:
        info = client.get_collection(COLLECTION_NAME)
        count = info.points_count or info.vectors_count or 0
    except Exception:
        count = 0

    if count > 0:
        log.info(f"Found {count} vectors. Skipping ingestion, rebuilding BM25 only...")
        _rebuild_bm25_only()
    else:
        log.info("No data found. Running full ingestion (first time)...")
        ingest_documents()


def rag_pipeline(query):
    context = recursive_search(query)
    best_score = max((c.get("rerank_score", 0) for c in context), default=0)

    if not context or best_score < 0.2:
        log.info("No strong local match in knowledge base — falling back to Google Search via Gemini")
        web_answer = get_web_grounded_answer(query)
        if web_answer:
            return {
                "query": query,
                "confidence": "Low (web search used)",
                "answer": web_answer,
                "retrieved_chunks": 0,
                "sources": [],
                "retrieval_details": [],
                "source_type": "web_search",
            }

    answer = get_llm_answer(query, context)

    if best_score > 0.5:
        confidence = "High"
    elif best_score > 0.2:
        confidence = "Medium"
    else:
        confidence = "Low"

    log.info(f"Confidence: {confidence} (best rerank score: {best_score:.4f})")

    sources = [{"page": c.get("page"), "file": c.get("source_file")} for c in context]
    unique_sources = []
    seen = set()
    for s in sources:
        key = (s["page"], s["file"])
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    return {
        "query": query,
        "confidence": confidence,
        "answer": answer,
        "retrieved_chunks": len(context),
        "sources": unique_sources,
        "retrieval_details": context,
        "source_type": "local",
    }


def rag_pipeline_stream(query):
    context = recursive_search(query)
    best_score = max((c.get("rerank_score", 0) for c in context), default=0)

    if not context or best_score < 0.2:
        log.info("No strong local match in knowledge base — falling back to Google Search via Gemini")
        web_answer = get_web_grounded_answer(query)
        if web_answer:
            return {"stream": None, "fallback": web_answer, "context": [], "source_type": "web_search"}
        log.warning("Web search fallback also unavailable.")
        return {
            "stream": None,
            "fallback": "No relevant information found in the knowledge base, and web search is currently unavailable.",
            "context": [],
            "source_type": "none",
        }

    context_text = "\n\n".join(c["text"][:1200] for c in context[:3])

    prompt = f"""
You are an expert medical assistant.

Answer the user's question using ONLY the provided context.

Instructions:
- Answer directly.
- Combine information from multiple passages.
- Do not copy the context.
- Ignore irrelevant information.
- Write naturally like ChatGPT.
- Use bullet points when useful.
- Give a complete answer.

Context:
{context_text}

Question:
{query}

Answer:
"""

    try:
        response = gemini_client.models.generate_content_stream(model="gemini-2.5-flash-lite", contents=prompt)
        return {"stream": response, "fallback": None, "context": context, "source_type": "local"}
    except Exception as e:
        log.warning(f"Gemini streaming error: {e}")
        fallback = "I found relevant information in the documents, but the language model is currently unavailable."
        return {"stream": None, "fallback": fallback, "context": context, "source_type": "local"}


#REPORT ANALYSIS FEATURE

def extract_report_text(file_bytes: bytes, filename: str) -> dict:
    filename_lower = filename.lower()

    if filename_lower.endswith(".txt"):
        log.info("Report type: plain text")
        try:
            text = file_bytes.decode("utf-8")
        except Exception:
            text = file_bytes.decode("latin-1")
        return {"text": text.strip(), "method": "plain_text"}

    if filename_lower.endswith((".jpg", ".jpeg", ".png", ".webp")):
        log.info("Report type: image — using EasyOCR")
        return _extract_text_from_image(file_bytes, filename_lower)

    if filename_lower.endswith(".pdf"):
        log.info("Report type: PDF — trying pypdf first")
        return _extract_text_from_pdf(file_bytes)

    log.warning(f"Unknown file type: {filename}")
    try:
        text = file_bytes.decode("utf-8", errors="ignore")
        return {"text": text.strip(), "method": "fallback_decode"}
    except Exception:
        return {"text": "", "method": "failed"}


def _extract_text_from_image(file_bytes: bytes, filename_lower: str) -> dict:
    """Local OCR using EasyOCR — no data sent to external servers."""
    try:
        import easyocr
        import numpy as np
        from PIL import Image
        import io

        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_array = np.array(image)

        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        result = reader.readtext(img_array, detail=0)

        text = "\n".join(result)
        log.info(f"EasyOCR extraction: {len(text)} characters extracted")
        return {"text": text, "method": "easyocr_local"}

    except Exception as e:
        log.error(f"EasyOCR extraction failed: {e}", exc_info=True)
        return {"text": "", "method": "failed"}


def _extract_text_from_pdf(file_bytes: bytes) -> dict:
    """Extract text from PDF using pypdf. Falls back to EasyOCR if scanned."""
    try:
        import io
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        text = text.strip()

        if len(text) < 100:
            log.info("PDF text too short — likely scanned. Switching to EasyOCR.")
            return _extract_text_from_image(file_bytes, "image.jpeg")

        log.info(f"PDF text extraction: {len(text)} characters extracted")
        return {"text": text, "method": "pypdf"}

    except Exception as e:
        log.error(f"PDF extraction failed: {e}", exc_info=True)
        return {"text": "", "method": "failed"}
    

def _simple_search(query: str, top_k=3) -> list:
    """
    Lightweight search — vector + BM25 + rerank only.
    Zero Gemini calls. Used during report analysis to save quota.
    """
    try:
        qdrant = get_client()
        query_embedding = model.encode(query, normalize_embeddings=True).tolist()

        vector_results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=10,
            with_payload=True
        ).points

        chunks = []
        for hit in vector_results:
            if hit.payload and hit.score > 0.3:
                chunks.append({
                    "chunk_id": hit.payload.get("chunk_id"),
                    "page": hit.payload.get("page", "Unknown"),
                    "source_file": hit.payload.get("source", "Unknown"),
                    "score": float(hit.score),
                    "text": hit.payload.get("text", "")
                })

        bm25_results = bm25_search(query, top_k=10)
        combined = {c["text"]: c for c in chunks + bm25_results}.values()
        combined = list(combined)

        if not combined:
            return []

        pairs = [(query, c["text"]) for c in combined]
        scores = reranker.predict(pairs)
        for c, s in zip(combined, scores):
            c["rerank_score"] = float(s)

        combined.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        filtered = [c for c in combined if c.get("rerank_score", 0) > 1.0]
        return filtered[:top_k] if filtered else []

    except Exception as e:
        log.warning(f"Simple search failed: {e}")
        return []    


def analyze_report_toa(report_text: str) -> dict:
    log.info("=== TOA LOOP STARTED ===")
    import time

    # ONE combined call instead of 3 separate calls
    log.info("TOA: Combined THOUGHT + OBSERVATION + SEARCH TERMS in one call")
    combined_prompt = f"""
You are a medical report analyst. This report may be a blood test, biopsy, pathology, radiology, or any other medical document.

Complete all 3 tasks below for the given report.

TASK 1 - Extract ALL clinically significant findings (numeric values, pathological findings, presence/absence findings, descriptive findings):
TASK 2 - Identify which findings are ABNORMAL and why.
TASK 3 - List search terms for the abnormal findings to look up in medical books.

Report:
{report_text}

Return in EXACTLY this format, no extra text:

PARAMETERS:
1. Parameter: <name> | Value: <value> | Unit: <unit or N/A> | Reference: <range or N/A>

ABNORMAL:
1. Parameter: <name> | Status: HIGH/LOW/PRESENT | Possible condition: <condition>

SEARCH_TERMS:
term1, term2, term3
"""

    combined_result = generate_with_retry("gemini-2.5-flash-lite", combined_prompt)
    log.info(f"Combined TOA result: {len(combined_result)} characters")

    # Parse the 3 sections from the single response
    thought_text = ""
    observation_text = ""
    conditions_raw = ""

    try:
        if "PARAMETERS:" in combined_result and "ABNORMAL:" in combined_result:
            params_section = combined_result.split("PARAMETERS:")[1].split("ABNORMAL:")[0].strip()
            abnormal_section = combined_result.split("ABNORMAL:")[1].split("SEARCH_TERMS:")[0].strip()
            search_section = combined_result.split("SEARCH_TERMS:")[1].strip() if "SEARCH_TERMS:" in combined_result else ""

            thought_text = params_section
            observation_text = abnormal_section
            conditions_raw = search_section
        else:
            log.warning("Combined prompt format not followed, using full result as thought")
            thought_text = combined_result
            observation_text = ""
            conditions_raw = ""
    except Exception as e:
        log.warning(f"Parsing combined result failed: {e}")
        thought_text = combined_result

    log.info(f"THOUGHT result: {len(thought_text)} characters")
    log.info(f"OBSERVATION result: {observation_text[:200]}")

    search_terms = [t.strip() for t in conditions_raw.split(",") if t.strip()]
    log.info(f"Search terms extracted: {search_terms}")

    # Book search — no query expansion/refinement to save calls
    book_findings = {}
    for term in search_terms[:5]:
        log.info(f"ACTION: searching books for '{term}'")
        try:
            chunks = _simple_search(term)  # uses vector + BM25 only, no Gemini calls
            if chunks:
                book_findings[term] = [
                    {
                        "text": c["text"][:800],
                        "page": c.get("page"),
                        "source_file": c.get("source_file"),
                        "rerank_score": c.get("rerank_score", 0)
                    }
                    for c in chunks
                ]
                log.info(f"Found {len(chunks)} book chunks for '{term}'")
            else:
                log.info(f"No book chunks found for '{term}' — trying Google Search fallback")
                web_answer = get_web_grounded_answer(term)
                if web_answer:
                    book_findings[term] = [
                        {
                            "text": web_answer[:800],
                            "page": "Web",
                            "source_file": "Google Search",
                            "rerank_score": 1.0
                        }
                    ]
                    log.info(f"Google Search fallback used for '{term}'")
        except Exception as e:
            log.warning(f"Book search failed for '{term}': {e}")
        time.sleep(1)

    log.info("=== TOA LOOP COMPLETE ===")

    return {
        "thought": thought_text,
        "observation": observation_text,
        "action": book_findings,
        "search_terms": search_terms
    }


def analyze_report_pipeline(file_bytes: bytes, filename: str) -> dict:
    log.info(f"=== REPORT ANALYSIS STARTED | file={filename} ===")

    extraction = extract_report_text(file_bytes, filename)
    report_text = extraction["text"]
    extraction_method = extraction["method"]

    log.info(f"Extraction method: {extraction_method} | text length: {len(report_text)}")

    if not report_text:
        log.error("Could not extract text from report")
        return {
            "success": False,
            "error": "Could not read the uploaded report. Please try a clearer image or a text-based PDF."
        }

    toa_result = analyze_report_toa(report_text)

    log.info("Generating final analysis from books + LLM")

    book_context_parts = []
    for term, chunks in toa_result["action"].items():
        for chunk in chunks[:2]:
            book_context_parts.append(f"[From medical books about {term}]:\n{chunk['text']}")

    book_context = "\n\n".join(book_context_parts) if book_context_parts else "No relevant book content found."

    final_prompt = f"""
You are an expert medical assistant analyzing a patient's medical report.

You have been given:
1. The extracted parameters from the report
2. Observations about abnormal values
3. Relevant information from medical textbooks

Your job is to give the patient a clear, complete, and helpful analysis.

--- EXTRACTED PARAMETERS ---
{toa_result['thought']}

--- ABNORMAL VALUES IDENTIFIED ---
{toa_result['observation']}

--- RELEVANT BOOK KNOWLEDGE ---
{book_context}

Instructions:
- Start with a brief summary of the report
- For each abnormal finding, explain:
  * What the parameter measures
  * Why the value is concerning
  * What condition it may indicate
  * What the medical books say about it
- Give practical recommendations and next steps
- Use simple language the patient can understand
- Use bullet points and clear headings
- At the end, add a disclaimer: "This analysis is for educational purposes only. Please consult a doctor."

Write the full analysis now:
"""

    final_analysis = generate_with_retry("gemini-2.5-flash-lite", final_prompt)
    if not final_analysis:
        final_analysis = "Analysis generation failed. Please try again."
    log.info(f"Final analysis generated: {len(final_analysis)} characters")

    sources = []
    seen = set()
    for term, chunks in toa_result["action"].items():
        for chunk in chunks:
            key = (chunk.get("page"), chunk.get("source_file"))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "page": chunk.get("page"),
                    "file": chunk.get("source_file"),
                    "term": term
                })

    log.info(f"=== REPORT ANALYSIS COMPLETE | sources={len(sources)} ===")

    return {
        "success": True,
        "filename": filename,
        "extraction_method": extraction_method,
        "report_text": report_text[:500] + "..." if len(report_text) > 500 else report_text,
        "toa": {
            "thought": toa_result["thought"],
            "observation": toa_result["observation"],
            "search_terms": toa_result["search_terms"]
        },
        "analysis": final_analysis,
        "sources": sources
    }