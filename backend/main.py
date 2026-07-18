from fastapi import FastAPI, Query, UploadFile, File, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import asyncio
import json
import logging
import os
import io
import time as _time
import httpx
from logging.handlers import TimedRotatingFileHandler
import rag
import page_index
import database
from auth import hash_password, verify_password, create_access_token, get_current_user
from patient_extraction import extract_patient_info, build_patient_context_block, PatientInfo
from pdf_export import generate_report_pdf
from fastapi.middleware.cors import CORSMiddleware
import guardrails
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import profiles_routes 


# LOGGER SETUP

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("medical_rag")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

file_handler = TimedRotatingFileHandler(
    filename=os.path.join(LOG_DIR, "app.log"),
    when="midnight",
    interval=1,
    backupCount=30,
    encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info("MAIN FILE LOADED")

#  CREATE APP FIRST
app = FastAPI()

#  THEN ADD CORS MIDDLEWARE (after app exists)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(profiles_routes.router)



_SCOPE_CACHE_TTL = 15
_SCOPE_CACHE = {}


# TOKEN / PAYWALL CONFIG

TOKEN_LIMIT = 50000
PAYWALL_MESSAGE = "You've hit your free limit of 50,000 tokens this month. Please upgrade to continue."


def token_limit_exceeded(user_id: int) -> bool:
    return database.get_token_usage(user_id) >= TOKEN_LIMIT


def get_scope(query: str, user_id: int) -> dict:
    key = (user_id, (query or "").strip().lower())
    cached = _SCOPE_CACHE.get(key)
    if cached and _time.time() - cached[1] < _SCOPE_CACHE_TTL:
        return cached[0]
    result = guardrails.check_query_scope(query, rag)
    _SCOPE_CACHE[key] = (result, _time.time())
    return result


def _finalize_charge(user_id: int, fallback_input: str = "", fallback_output: str = ""):
    """
    Reads whatever real usage got accumulated in rag's tracker during this
    request (covers query expansion, refinement, RAG generation, page
    index synthesis, web-grounded search — anything routed through
    rag.generate_with_retry / rag.generate_with_groq / rag.get_web_grounded_answer).

    Falls back to the old char/4 estimate only if nothing real got
    recorded — e.g. every LLM call failed outright, or a code path
    doesn't call an LLM at all (input/output are just facts, no cost).
    """
    usage = rag.get_tracked_usage()
    total = usage.get("total_tokens", 0)

    if total > 0:
        database.add_tokens(user_id, total)
        logger.info(f"Charged real usage | user={user_id} | prompt={usage['prompt_tokens']} completion={usage['completion_tokens']} total={total}")
    else:
        estimated = database.estimate_tokens(fallback_input) + database.estimate_tokens(fallback_output)
        if estimated > 0:
            database.add_tokens(user_id, estimated)
            logger.info(f"Charged ESTIMATED usage (no real usage captured) | user={user_id} | tokens={estimated}")


CHECK_EVERY = 20


class DisconnectChecker:
    """Wraps request.is_disconnected() with a counter so we only actually
    check every `every` calls, keeping the hot loop cheap."""

    def __init__(self, request: Request, every: int = CHECK_EVERY):
        self.request = request
        self.every = every
        self._n = 0
        self.disconnected = False

    async def check(self) -> bool:
        if self.disconnected:
            return True
        self._n += 1
        if self._n % self.every != 0:
            return False
        self.disconnected = await self.request.is_disconnected()
        return self.disconnected


def _try_close_stream(stream_obj):
    """Best-effort: ask the Gemini/Groq stream object to close its
    underlying connection so we stop pulling (and paying for) further
    chunks once the client has disconnected. Safe no-op if unsupported."""
    close_fn = getattr(stream_obj, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            pass


scheduler = AsyncIOScheduler()


def _monthly_token_reset_job():
    try:
        count = database.reset_all_tokens()
        logger.info(f"Monthly token reset complete | users_reset={count}")
    except Exception as e:
        logger.error(f"Monthly token reset failed: {e}", exc_info=True)


@app.on_event("startup")
def startup():
    logger.info("INITIALIZING DATABASE...")
    database.init_db()
    logger.info("DATABASE READY")
    logger.info("INITIALIZING RAG...")
    rag.initialize()
    logger.info("RAG READY")

    scheduler.add_job(
        _monthly_token_reset_job,
        trigger=CronTrigger(day=1, hour=0, minute=0),
        id="monthly_token_reset",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Scheduler started | monthly token reset set for 1st @ 00:00")


@app.on_event("shutdown")
def shutdown():
    scheduler.shutdown()


@app.get("/")
def home():
    logger.info("GET / - health check")
    return {"message": "Medical RAG Backend Running"}


# AUTH

class SignupRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/signup")
def signup(req: SignupRequest):
    logger.info(f"POST /signup | username={req.username}")
    if len(req.username.strip()) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters.")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    existing = database.get_user_by_username(req.username)
    if existing:
        raise HTTPException(status_code=400, detail="That username is already taken.")

    password_hash = hash_password(req.password)
    new_id = database.create_user(req.username, password_hash)
    logger.info(f"User created | username={req.username}")

    database.log_activity(new_id, "signup", {"username": req.username})

    token = create_access_token(req.username)
    return {"access_token": token, "token_type": "bearer", "username": req.username}


@app.post("/login")
def login(req: LoginRequest):
    logger.info(f"POST /login | username={req.username}")
    user = database.get_user_by_username(req.username)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password.")

    database.log_activity(user["id"], "login")

    token = create_access_token(req.username)
    return {"access_token": token, "token_type": "bearer", "username": req.username}


@app.get("/me")
def me(current_user: dict = Depends(get_current_user)):
    tokens_used = database.get_token_usage(current_user["id"])
    return {
        "username": current_user["username"],
        "tokens_used": tokens_used,
        "tokens_limit": TOKEN_LIMIT,
        "tokens_remaining": max(0, TOKEN_LIMIT - tokens_used),
    }


@app.get("/ingest")
def ingest():
    logger.info("GET /ingest - ingesting documents")
    vectors = rag.ingest_documents()
    logger.info(f"Ingestion complete | vectors_stored={vectors}")
    return {"vectors_stored": vectors}


@app.post("/chat")
def chat(query: str = Query(...), current_user: dict = Depends(get_current_user)):
    logger.info(f"POST /chat | user={current_user['username']} | query={query!r}")

    scope = get_scope(query, current_user["id"])
    if not scope["in_scope"]:
        logger.info(f"POST /chat | off-topic blocked | user={current_user['username']} | query={query!r}")
        database.log_activity(current_user["id"], "off_topic_rejected", {
            "query": query,
            "endpoint": "chat",
        })
        return {"answer": guardrails.OFF_TOPIC_MESSAGE, "off_topic": True, "context": [], "sources": []}

    if token_limit_exceeded(current_user["id"]):
        logger.info(f"POST /chat | paywall blocked | user={current_user['username']}")
        return {"answer": PAYWALL_MESSAGE, "paywall": True, "context": [], "sources": []}

    rag.start_usage_tracking()
    result = rag.rag_pipeline(query)
    logger.info("POST /chat - response generated")

    _finalize_charge(current_user["id"], fallback_input=query, fallback_output=result.get("answer", ""))

    return result


@app.post("/chat-stream")
def chat_stream(query: str = Query(...), request: Request = None, current_user: dict = Depends(get_current_user)):
    logger.info(f"POST /chat-stream | user={current_user['username']} | query={query!r}")
    user_id = current_user["id"]

    scope = get_scope(query, user_id)
    if not scope["in_scope"]:
        logger.info(f"POST /chat-stream | off-topic blocked | user={current_user['username']} | query={query!r}")

        def generate_blocked():
            import time
            for char in guardrails.OFF_TOPIC_MESSAGE:
                yield f"data: {json.dumps({'type':'text','content':char})}\n\n"
                time.sleep(0.005)
            database.log_activity(user_id, "off_topic_rejected", {
                "query": query,
                "endpoint": "chat-stream",
            })
            yield f"data: {json.dumps({'type':'done','sources':[],'web_sources':[],'retrieval_details':[],'source_type':'blocked'})}\n\n"

        return StreamingResponse(generate_blocked(), media_type="text/event-stream")

    if token_limit_exceeded(user_id):
        logger.info(f"POST /chat-stream | paywall blocked | user={current_user['username']}")

        def generate_paywall():
            for char in PAYWALL_MESSAGE:
                yield f"data: {json.dumps({'type':'text','content':char})}\n\n"
                _time.sleep(0.005)
            yield f"data: {json.dumps({'type':'done','sources':[],'web_sources':[],'retrieval_details':[],'source_type':'paywall'})}\n\n"

        return StreamingResponse(generate_paywall(), media_type="text/event-stream")

    async def generate():
        context = []
        source_type = "local"
        web_sources = []
        full_response_text = ""
        checker = DisconnectChecker(request)

        rag.start_usage_tracking()

        try:
            result = rag.rag_pipeline_stream(query)
            response = result["stream"]
            context = result["context"]
            fallback = result["fallback"]
            source_type = result.get("source_type", "local")
            web_sources = result.get("web_sources", [])

            if fallback:
                logger.warning("Using fallback response (no stream returned)")
                full_response_text = fallback
                for char in fallback:
                    if await checker.check():
                        break
                    yield f"data: {json.dumps({'type':'text','content':char})}\n\n"
                    await asyncio.sleep(0.005)

            elif response:
                logger.debug("Streaming response chunks to client")
                last_usage = None
                for chunk in response:
                    if await checker.check():
                        _try_close_stream(response)
                        break
                    if getattr(chunk, "usage_metadata", None):
                        last_usage = chunk.usage_metadata
                    if chunk.text:
                        full_response_text += chunk.text
                        for char in chunk.text:
                            if await checker.check():
                                _try_close_stream(response)
                                break
                            yield f"data: {json.dumps({'type':'text','content':char})}\n\n"
                            await asyncio.sleep(0.005)
                    if checker.disconnected:
                        break

                # Gemini attaches cumulative usage progressively; the last
                # chunk seen holds the final total for this stream.
                if last_usage:
                    rag.record_external_usage(
                        prompt_tokens=getattr(last_usage, "prompt_token_count", 0) or 0,
                        completion_tokens=getattr(last_usage, "candidates_token_count", 0) or 0,
                    )

        except Exception:
            logger.error("Exception during chat-stream generation", exc_info=True)
            logger.warning("Gemini stream failed — trying Groq fallback...")
            try:
                context_text = "\n\n".join(c.get("text", "")[:1200] for c in context[:3])
                groq_stream = rag.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{
                        "role": "user",
                        "content": f"You are a medical assistant. Answer this question using the context below.\n\nContext:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
                    }],
                    stream=True,
                    max_tokens=1024,
                    stream_options={"include_usage": True},
                )
                last_groq_usage = None
                for chunk in groq_stream:
                    if await checker.check():
                        _try_close_stream(groq_stream)
                        break
                    if getattr(chunk, "usage", None):
                        last_groq_usage = chunk.usage
                    if not chunk.choices:
                        # final usage-only chunk has no choices — nothing to stream
                        continue
                    text = chunk.choices[0].delta.content or ""
                    full_response_text += text
                    for char in text:
                        if await checker.check():
                            _try_close_stream(groq_stream)
                            break
                        yield f"data: {json.dumps({'type':'text','content':char})}\n\n"
                        await asyncio.sleep(0.005)
                    if checker.disconnected:
                        break

                if last_groq_usage:
                    rag.record_external_usage(
                        prompt_tokens=getattr(last_groq_usage, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(last_groq_usage, "completion_tokens", 0) or 0,
                    )

                if not checker.disconnected:
                    logger.info("Groq stream fallback succeeded")
            except Exception as groq_err:
                logger.error(f"Groq fallback also failed: {groq_err}")
                msg = "Both Gemini and Groq are temporarily unavailable. Please try again in a minute."
                for char in msg:
                    if await checker.check():
                        break
                    yield f"data: {json.dumps({'type':'text','content':char})}\n\n"
                    await asyncio.sleep(0.005)

        if checker.disconnected:
            logger.info(f"POST /chat-stream | client disconnected mid-stream | user={current_user['username']} | query={query!r}")
            return  # no token charge, no activity log entry for a stopped query

        sources = []
        seen = set()
        for c in context:
            key = (c["page"], c["source_file"])
            if key not in seen:
                seen.add(key)
                sources.append({"page": c["page"], "file": c["source_file"]})

        _finalize_charge(user_id, fallback_input=query, fallback_output=full_response_text)

        database.log_activity(user_id, "query", {
            "query": query,
            "answer": full_response_text,
            "source_type": source_type,
            "sources": sources,
            "web_sources": web_sources,
        })

        logger.info(f"Stream complete | rag_sources={len(sources)} | rag_web={len(web_sources)}")
        yield f"data: {json.dumps({'type':'done','sources':sources,'web_sources':web_sources,'retrieval_details':context,'source_type':source_type})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/page-index-stream")
def page_index_stream(query: str = Query(...), request: Request = None, current_user: dict = Depends(get_current_user)):
    """
    SEPARATE Page Index pipeline - tree-based retrieval with reasoning.
    Returns: Full answer + hierarchical sources.
    """
    logger.info(f"POST /page-index-stream | user={current_user['username']} | query={query!r}")
    user_id = current_user["id"]

    scope = get_scope(query, user_id)
    if not scope["in_scope"]:
        logger.info(f"POST /page-index-stream | off-topic blocked | user={current_user['username']} | query={query!r}")

        def generate_blocked():
            import time
            for char in guardrails.OFF_TOPIC_MESSAGE:
                yield f"data: {json.dumps({'type':'text','content':char})}\n\n"
                time.sleep(0.005)
            database.log_activity(user_id, "off_topic_rejected", {
                "query": query,
                "endpoint": "page-index-stream",
            })
            yield f"data: {json.dumps({'type':'done','sources':[],'web_sources':[],'retrieval_details':[],'source_type':'blocked'})}\n\n"

        return StreamingResponse(generate_blocked(), media_type="text/event-stream")

    if token_limit_exceeded(user_id):
        logger.info(f"POST /page-index-stream | paywall blocked | user={current_user['username']}")

        def generate_paywall():
            for char in PAYWALL_MESSAGE:
                yield f"data: {json.dumps({'type':'text','content':char})}\n\n"
                _time.sleep(0.005)
            yield f"data: {json.dumps({'type':'done','page_index_sources':{}, 'source_type':'paywall'})}\n\n"

        return StreamingResponse(generate_paywall(), media_type="text/event-stream")

    async def generate():
        answer = ""
        sources = {}
        checker = DisconnectChecker(request)

        rag.start_usage_tracking()
        # NOTE: page_index.page_index_pipeline is in a separate module I
        # haven't seen. If it calls rag.generate_with_retry / rag.generate_with_groq
        # internally, its usage is captured automatically here same as
        # everything else. If it calls gemini_client/groq_client directly
        # instead, this will undercount — worth checking page_index.py.

        try:
            if await checker.check():
                return
            result = page_index.page_index_pipeline(query)

            answer = result.get("answer", "")
            sources = result.get("sources", {})

            for char in answer:
                if await checker.check():
                    break
                yield f"data: {json.dumps({'type':'text','content':char})}\n\n"
                await asyncio.sleep(0.003)

        except Exception as e:
            logger.error(f"Page index pipeline failed: {e}", exc_info=True)
            msg = "Page index retrieval failed. Please try again."
            answer = msg
            for char in msg:
                if await checker.check():
                    break
                yield f"data: {json.dumps({'type':'text','content':char})}\n\n"
                await asyncio.sleep(0.005)

        if checker.disconnected:
            logger.info(f"POST /page-index-stream | client disconnected mid-stream | user={current_user['username']} | query={query!r}")
            return  # no token charge, no activity log entry for a stopped query

        _finalize_charge(user_id, fallback_input=query, fallback_output=answer)

        database.log_activity(user_id, "page_index_query", {
            "query": query,
            "answer": answer,
            "sources": sources,
        })

        logger.info(f"Page index stream complete")
        yield f"data: {json.dumps({'type':'done','page_index_sources':sources})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# NON-STREAMING (keep for fallback)
@app.post("/analyze-report")
async def analyze_report(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    logger.info(f"POST /analyze-report | user={current_user['username']} | filename={file.filename} | content_type={file.content_type}")

    if token_limit_exceeded(current_user["id"]):
        logger.info(f"POST /analyze-report | paywall blocked | user={current_user['username']}")
        return {"success": False, "paywall": True, "error": PAYWALL_MESSAGE}

    rag.start_usage_tracking()
    try:
        file_bytes = await file.read()
        logger.info(f"File received | size={len(file_bytes)} bytes")
        result = rag.analyze_report_pipeline(file_bytes, file.filename)
        if result.get("success"):
            logger.info(f"Report analysis complete | sources={len(result.get('sources', []))}")
        else:
            logger.warning(f"Report analysis failed | error={result.get('error')}")

        _finalize_charge(
            current_user["id"],
            fallback_input=result.get("report_text", ""),
            fallback_output=result.get("analysis", ""),
        )

        return result
    except Exception as e:
        logger.error(f"Unexpected error in /analyze-report | error={e}", exc_info=True)
        return {"success": False, "error": f"Server error: {str(e)}"}


# STEP 1 — extract report text + patient info, no analysis yet
@app.post("/extract-patient-info")
async def extract_patient_info_endpoint(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    logger.info(f"POST /extract-patient-info | user={current_user['username']} | filename={file.filename}")

    if token_limit_exceeded(current_user["id"]):
        logger.info(f"POST /extract-patient-info | paywall blocked | user={current_user['username']}")
        return {"success": False, "paywall": True, "error": PAYWALL_MESSAGE}

    rag.start_usage_tracking()
    # NOTE: extract_patient_info() lives in patient_extraction.py, which I
    # haven't seen. If it calls gemini_client directly (not through
    # rag.generate_with_retry), its usage won't be captured here and this
    # will silently fall back to the char/4 estimate. Paste patient_extraction.py
    # if you want this fully accurate too.

    try:
        file_bytes = await file.read()
        extraction = rag.extract_report_text(file_bytes, file.filename)
        report_text = extraction["text"]
        extraction_method = extraction["method"]

        if not report_text:
            logger.warning("Could not extract text from uploaded report")
            database.log_activity(current_user["id"], "upload", {"filename": file.filename}, status="error", error_msg="Could not extract text from report")
            return {"success": False, "error": "Could not read the report. Try a clearer image or text-based PDF."}

        patient_info = extract_patient_info(report_text, gemini_client=rag.gemini_client)
        logger.info(f"Patient info extracted | name={patient_info.name} | age={patient_info.age}")

        _finalize_charge(current_user["id"], fallback_input=report_text, fallback_output=str(patient_info.dict()))

        database.log_activity(current_user["id"], "upload", {"filename": file.filename, "method": extraction_method})

        return {
            "success": True,
            "report_text": report_text,
            "extraction_method": extraction_method,
            "patient_info": patient_info.dict(),
        }
    except Exception as e:
        logger.error(f"Unexpected error in /extract-patient-info | error={e}", exc_info=True)
        database.log_activity(current_user["id"], "upload", {"filename": file.filename}, status="error", error_msg=str(e))
        return {"success": False, "error": f"Server error: {str(e)}"}


class AnalyzeRequest(BaseModel):
    report_text: str
    patient_info: PatientInfo
    extraction_method: Optional[str] = None
    filename: Optional[str] = None


# STREAMING REPORT ANALYSIS
@app.post("/analyze-report-stream")
def analyze_report_stream(req: AnalyzeRequest, request: Request, current_user: dict = Depends(get_current_user)):
    logger.info(f"POST /analyze-report-stream | user={current_user['username']} | patient={req.patient_info.name} age={req.patient_info.age}")

    report_text = req.report_text
    extraction_method = req.extraction_method or "cached"
    patient_context = build_patient_context_block(req.patient_info)
    user_id = current_user["id"]
    filename = req.filename or "report"

    if token_limit_exceeded(user_id):
        logger.info(f"POST /analyze-report-stream | paywall blocked | user={current_user['username']}")

        def generate_paywall():
            for char in PAYWALL_MESSAGE:
                yield f"data: {json.dumps({'type':'text','content':char})}\n\n"
                _time.sleep(0.005)
            yield f"data: {json.dumps({'type':'done','sources':[],'extraction_method':extraction_method,'report_id':None,'source_type':'paywall'})}\n\n"

        return StreamingResponse(generate_paywall(), media_type="text/event-stream")

    async def generate():
        checker = DisconnectChecker(request)
        rag.start_usage_tracking()

        yield f"data: {json.dumps({'type':'status','content':'Identifying parameters...'})}\n\n"

        toa_result = rag.analyze_report_toa(report_text)

        if await checker.check():
            return

        yield f"data: {json.dumps({'type':'toa','thought': toa_result['thought'], 'observation': toa_result['observation'], 'search_terms': toa_result['search_terms']})}\n\n"

        yield f"data: {json.dumps({'type':'status','content':'Generating analysis...'})}\n\n"

        book_context_parts = []
        for term, chunks in toa_result["action"].items():
            for chunk in chunks[:2]:
                book_context_parts.append(f"[From medical books about {term}]:\n{chunk['text']}")

        book_context = "\n\n".join(book_context_parts) if book_context_parts else "No relevant book content found."

        final_prompt = f"""
You are a clinical medical assistant writing a structured diagnostic report for a patient.

{patient_context}

--- EXTRACTED PARAMETERS ---
{toa_result['thought']}

--- ABNORMAL VALUES ---
{toa_result['observation']}

--- RELEVANT MEDICAL KNOWLEDGE ---
{book_context}

Write a professional medical analysis report with the following sections:

## Clinical Summary
3-4 sentences. Describe the patient's overall health picture based on the report. Mention which systems are affected.

## Findings & Interpretation
For each abnormal parameter:
- **[Parameter Name]** — Value: [X] (Normal: [Y])
  What this measures, why this value is abnormal for this patient's age/sex, and what condition it may suggest. Reference medical knowledge where relevant. Keep to 3-4 sentences per finding.

## Clinical Recommendations
- Specific follow-up tests or specialist referrals needed
- Lifestyle or dietary changes relevant to the findings
- Any urgency level (routine / soon / urgent)
Tailor recommendations to the patient's age, sex, and any allergies mentioned.

## Important Note
This report is generated by an AI system for informational purposes only and does not replace a consultation with a qualified medical professional. Please share this report with your doctor.

Use clear, professional language that a patient can understand. Avoid unnecessary repetition.
"""

        full_analysis = ""
        max_stream_retries = 3

        for stream_attempt in range(max_stream_retries):
            if checker.disconnected:
                break
            try:
                response = rag.gemini_client.models.generate_content_stream(
                    model="gemini-2.5-flash",
                    contents=final_prompt
                )
                full_analysis_parts = []
                last_usage = None
                for chunk in response:
                    if await checker.check():
                        _try_close_stream(response)
                        break
                    if getattr(chunk, "usage_metadata", None):
                        last_usage = chunk.usage_metadata
                    if chunk.text:
                        full_analysis_parts.append(chunk.text)
                        for char in chunk.text:
                            if await checker.check():
                                _try_close_stream(response)
                                break
                            yield f"data: {json.dumps({'type':'text','content':char})}\n\n"
                            await asyncio.sleep(0.003)
                    if checker.disconnected:
                        break
                full_analysis = "".join(full_analysis_parts)

                if last_usage:
                    rag.record_external_usage(
                        prompt_tokens=getattr(last_usage, "prompt_token_count", 0) or 0,
                        completion_tokens=getattr(last_usage, "candidates_token_count", 0) or 0,
                    )
                break

            except Exception as e:
                error_str = str(e).lower()
                logger.error(f"Streaming analysis failed (attempt {stream_attempt+1}): {e}", exc_info=True)
                if ("503" in error_str or "unavailable" in error_str or "429" in error_str or "quota" in error_str) and stream_attempt < max_stream_retries - 1:
                    wait = (2 ** stream_attempt) * 5
                    logger.warning(f"Retrying stream in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    logger.warning("Stream failed, falling back to generate_with_retry (Gemini → Groq)")
                    full_analysis = rag.generate_with_retry("gemini-2.5-flash", final_prompt)
                    if full_analysis:
                        yield f"data: {json.dumps({'type':'clear'})}\n\n"
                        for char in full_analysis:
                            if await checker.check():
                                break
                            yield f"data: {json.dumps({'type':'text','content':char})}\n\n"
                            await asyncio.sleep(0.003)
                    else:
                        yield f"data: {json.dumps({'type':'text','content':'Analysis generation failed. Please try again later.'})}\n\n"
                    break

        if checker.disconnected:
            logger.info(f"POST /analyze-report-stream | client disconnected mid-stream | user={current_user['username']}")
            return  # no token charge, no report saved, no activity log entry

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
                        "term": term,
                        "url": chunk.get("url"),
                    })

        _finalize_charge(user_id, fallback_input=report_text, fallback_output=full_analysis)

        try:
            report_id = database.save_report(
                user_id=user_id,
                filename=filename,
                patient_info=req.patient_info.dict(),
                thought=toa_result["thought"],
                observation=toa_result["observation"],
                search_terms=toa_result["search_terms"],
                analysis=full_analysis,
                sources=sources,
                extraction_method=extraction_method,
            )
            logger.info(f"Report saved | report_id={report_id} | user={current_user['username']}")
            database.log_activity(user_id, "report_generated", {"report_id": report_id, "filename": filename})
        except Exception as e:
            logger.error(f"Failed to save report to database: {e}", exc_info=True)
            database.log_activity(user_id, "report_generated", {"filename": filename}, status="error", error_msg=str(e))
            report_id = None

        yield f"data: {json.dumps({'type':'done','sources':sources,'extraction_method':extraction_method,'report_id':report_id})}\n\n"
        logger.info(f"Report stream complete | sources={len(sources)}")

    return StreamingResponse(generate(), media_type="text/event-stream")


# REPORT HISTORY

@app.get("/my-reports")
def my_reports(current_user: dict = Depends(get_current_user)):
    logger.info(f"GET /my-reports | user={current_user['username']}")
    reports = database.get_reports_for_user(current_user["id"])
    return {"reports": reports}


@app.get("/my-reports/{report_id}")
def my_report_detail(report_id: int, current_user: dict = Depends(get_current_user)):
    logger.info(f"GET /my-reports/{report_id} | user={current_user['username']}")
    report = database.get_report_by_id(report_id, current_user["id"])
    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")
    return report


@app.get("/my-reports/{report_id}/download")
def download_report(report_id: int, current_user: dict = Depends(get_current_user)):
    logger.info(f"GET /my-reports/{report_id}/download | user={current_user['username']}")
    report = database.get_report_by_id(report_id, current_user["id"])
    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")

    pdf_bytes = generate_report_pdf(report)
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=report_{report_id}.pdf"},
    )


@app.delete("/my-reports/{report_id}")
def delete_report(report_id: int, current_user: dict = Depends(get_current_user)):
    logger.info(f"DELETE /my-reports/{report_id} | user={current_user['username']}")
    report = database.get_report_by_id(report_id, current_user["id"])
    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")
    database.delete_report(report_id, current_user["id"])
    database.log_activity(current_user["id"], "report_deleted", {"report_id": report_id})
    return {"success": True, "message": f"Report {report_id} deleted."}


@app.get("/my-reports/{report_id}/preview")
def preview_report(report_id: int, current_user: dict = Depends(get_current_user)):
    logger.info(f"GET /my-reports/{report_id}/preview | user={current_user['username']}")
    report = database.get_report_by_id(report_id, current_user["id"])
    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")
    return report


@app.get("/my-activity")
def my_activity(current_user: dict = Depends(get_current_user)):
    logger.info(f"GET /my-activity | user={current_user['username']}")
    logs    = database.get_activity_for_user(current_user["id"], limit=100)
    summary = database.get_activity_summary_for_user(current_user["id"])
    return {"logs": logs, "summary": summary}


# NEARBY DOCTORS

def get_city_from_coords(lat: float, lng: float) -> str:
    """Use OpenStreetMap Nominatim (free, no API key) to reverse geocode lat/lng → city name."""
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lng}&format=json"
        headers = {"User-Agent": "MedicalRAGAssistant/1.0"}
        response = httpx.get(url, headers=headers, timeout=5)
        data = response.json()
        address = data.get("address", {})
        city = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("county")
            or address.get("state")
            or "your area"
        )
        state = address.get("state", "")
        logger.info(f"Reverse geocode | lat={lat} lng={lng} → city={city}, state={state}")
        return city, state
    except Exception as e:
        logger.warning(f"Nominatim reverse geocode failed: {e}")
        return "your area", ""


def extract_specialist_from_query(query: str) -> str:
    """Figure out what type of doctor to search for based on the user's query."""
    query_lower = query.lower()

    specialist_map = {
        "cardiologist": ["heart", "cardiac", "cardio", "chest pain", "ecg", "blood pressure", "hypertension"],
        "neurologist": ["brain", "neuro", "headache", "migraine", "seizure", "stroke", "nerve"],
        "orthopedic": ["bone", "joint", "fracture", "spine", "back pain", "knee", "shoulder", "orthopedic"],
        "dermatologist": ["skin", "rash", "acne", "eczema", "psoriasis", "dermat"],
        "gastroenterologist": ["stomach", "liver", "digestive", "gut", "ibs", "acid", "gastro", "colon"],
        "pulmonologist": ["lung", "breathing", "asthma", "copd", "respiratory", "cough", "pulmon"],
        "endocrinologist": ["diabetes", "thyroid", "hormone", "insulin", "sugar", "endocrin"],
        "oncologist": ["cancer", "tumor", "oncolog", "chemotherapy", "biopsy"],
        "psychiatrist": ["mental health", "depression", "anxiety", "psychiatr", "therapy"],
        "gynecologist": ["gynecol", "women health", "pregnancy", "uterus", "ovary", "periods"],
        "pediatrician": ["child", "baby", "infant", "pediatr", "kids"],
        "urologist": ["kidney", "urology", "bladder", "urinary", "prostate"],
        "ophthalmologist": ["eye", "vision", "ophthalmol", "optometrist"],
        "ent specialist": ["ear", "nose", "throat", "ent", "sinus", "hearing"],
        "general physician": ["fever", "cold", "flu", "general", "doctor near", "nearby doctor", "physician", "clinic"],
    }

    for specialist, keywords in specialist_map.items():
        for kw in keywords:
            if kw in query_lower:
                return specialist

    return "general physician"


@app.get("/nearby-doctors")
async def nearby_doctors(
    request: Request,
    lat: float = Query(...),
    lng: float = Query(...),
    query: str = Query(...),
    current_user: dict = Depends(get_current_user)
):
    scope = get_scope(query, current_user["id"])
    if not scope["in_scope"]:
        logger.info(f"GET /nearby-doctors | off-topic blocked | user={current_user['username']} | query={query!r}")
        database.log_activity(current_user["id"], "off_topic_rejected", {
            "query": query,
            "endpoint": "nearby-doctors",
        })
        return {"success": False, "off_topic": True, "info": guardrails.OFF_TOPIC_MESSAGE}

    if token_limit_exceeded(current_user["id"]):
        logger.info(f"GET /nearby-doctors | paywall blocked | user={current_user['username']}")
        return {"success": False, "off_topic": False, "paywall": True, "info": PAYWALL_MESSAGE}

    if await request.is_disconnected():
        logger.info(f"GET /nearby-doctors | client already disconnected | user={current_user['username']}")
        return {"success": False, "off_topic": False, "info": "Request cancelled."}

    logger.info(
        f"GET /nearby-doctors | user={current_user['username']} | lat={lat} lng={lng} | query={query!r}"
    )

    city, state = get_city_from_coords(lat, lng)
    specialist = extract_specialist_from_query(query)
    location_str = ", ".join([x for x in [city, state] if x])

    if not city:
        return {
            "success": False,
            "specialist": specialist,
            "location": location_str,
            "city": city,
            "info": "Could not determine city from coordinates."
        }

    search_prompt = f"""Find {specialist} near {location_str}.

List 4-5 real clinics or doctors in or near {location_str}. For each one provide:
- Doctor/Clinic Name
- Address
- Contact Number
- Speciality

Only return the list. No advice, no explanations, no disclaimers. Keep it short and factual."""

    logger.info(f"Searching for {specialist} near {location_str}")
    logger.info(f"Prompt={search_prompt}")

    rag.start_usage_tracking()

    try:
        doctor_result = rag.get_web_grounded_answer(search_prompt)
        logger.info(f"doctor_result={doctor_result!r}")

        doctor_info = doctor_result.get("answer", "") if isinstance(doctor_result, dict) else ""

        if not str(doctor_info).strip():
            doctor_info = (
                f"Could not fetch doctor information right now. "
                f"Please search for '{specialist} near {location_str}' on Google Maps for the most accurate results."
            )

    except Exception as e:
        logger.error(f"Doctor search failed: {e}", exc_info=True)
        doctor_info = (
            f"Could not fetch doctor information. "
            f"Please search for '{specialist} near {location_str}' on Google Maps."
        )

    if await request.is_disconnected():
        logger.info(f"GET /nearby-doctors | client disconnected before logging | user={current_user['username']}")
        return {"success": False, "off_topic": False, "info": "Request cancelled."}

    _finalize_charge(current_user["id"], fallback_input=search_prompt, fallback_output=doctor_info)

    database.log_activity(current_user["id"], "nearby_doctors", {
        "query": query,
        "specialist": specialist,
        "city": city,
        "lat": lat,
        "lng": lng,
        "info": doctor_info,
    })

    return {
        "success": True,
        "specialist": specialist,
        "location": location_str,
        "city": city,
        "info": doctor_info,
    }

# ACTIVITY LOG ENDPOINT
@app.get("/activity-log")
async def get_activity_log(current_user: dict = Depends(get_current_user)):
    """
    Get all activities for the logged-in user.
    Reads from user_activity_logs via database.get_activity_for_user,
    then flattens the metadata JSON into the fields the frontend expects.
    """
    username = current_user["username"]

    try:
        rows = database.get_activity_for_user(current_user["id"], limit=200)

        activities = []
        for row in rows:
            meta = row.get("metadata") or {}
            activities.append({
                "id": row["id"],
                "username": username,
                "action": row["action_type"],
                "query": meta.get("query"),
                "filename": meta.get("filename"),
                "patient_info": meta.get("patient_info"),
                "result": row.get("error_msg") if row.get("status") == "error" else None,
                "status": row.get("status"),
                "metadata": meta,
                "timestamp": row["created_at"].isoformat() if row.get("created_at") else None,
            })

        return {
            "success": True,
            "username": username,
            "activities": activities,
            "count": len(activities)
        }

    except Exception as e:
        logger.error(f"Error fetching activity log for {username}: {e}")
        return {
            "success": False,
            "error": "Could not fetch activity log",
            "activities": []
        }

# MCP Integration
from fastapi_mcp import FastApiMCP

mcp = FastApiMCP(app)
mcp.mount()