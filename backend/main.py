from fastapi import FastAPI, Query, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json 
import logging 
import 
import io
from logging.handlers import TimedRotatingFileHandler
import rag
import database
from auth import hash_password, verify_password, create_access_token, get_current_user
from patient_extraction import extract_patient_info, build_patient_context_block, PatientInfo
from pdf_export import generate_report_pdf


#LOGGER SETUP

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

app = FastAPI()

@app.on_event("startup")
def startup():
    logger.info("INITIALIZING DATABASE...")
    database.init_db()
    logger.info("DATABASE READY")
    logger.info("INITIALIZING RAG...")
    rag.initialize()
    logger.info("RAG READY")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    logger.info("GET / - health check")
    return {"message": "Medical RAG Backend Running"}


#AUTH

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
    database.create_user(req.username, password_hash)
    logger.info(f"User created | username={req.username}")

    token = create_access_token(req.username)
    return {"access_token": token, "token_type": "bearer", "username": req.username}


@app.post("/login")
def login(req: LoginRequest):
    logger.info(f"POST /login | username={req.username}")
    user = database.get_user_by_username(req.username)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password.")

    token = create_access_token(req.username)
    return {"access_token": token, "token_type": "bearer", "username": req.username}


@app.get("/me")
def me(current_user: dict = Depends(get_current_user)):
    return {"username": current_user["username"]}

@app.get("/build-db")
def create_db():
    logger.info("GET /build-db - building vector database")
    try:
        chunks = rag.build_vector_db()
        logger.info(f"Vector DB built successfully | chunks_created={chunks}")
        return {"status": "success", "chunks_created": chunks}
    except Exception as e:
        logger.error(f"Vector DB build failed | error={e}", exc_info=True)
        return {"status": "failed", "error": str(e)}

@app.get("/create-collection")
def create_collection():
    logger.info("GET /create-collection")
    result = rag.create_qdrant_collection()
    logger.info(f"Collection creation result: {result}")
    return {"status": result}

@app.get("/test-embedding")
def test_embedding():
    logger.info("GET /test-embedding")
    size = rag.generate_embeddings()
    logger.info(f"Embedding size: {size}")
    return {"embedding_size": size}

@app.get("/ingest")
def ingest():
    logger.info("GET /ingest - ingesting documents")
    vectors = rag.ingest_documents()
    logger.info(f"Ingestion complete | vectors_stored={vectors}")
    return {"vectors_stored": vectors}

@app.post("/chat")
def chat(query: str = Query(...), current_user: dict = Depends(get_current_user)):
    logger.info(f"POST /chat | user={current_user['username']} | query={query!r}")
    result = rag.rag_pipeline(query)
    logger.info("POST /chat - response generated")
    return result

@app.post("/chat-stream")
def chat_stream(query: str = Query(...), current_user: dict = Depends(get_current_user)):
    logger.info(f"POST /chat-stream | user={current_user['username']} | query={query!r}")

    def generate():
        import time
        import traceback
        context = []
        source_type = "local"

        try:
            result = rag.rag_pipeline_stream(query)
            response = result["stream"]
            context = result["context"]
            fallback = result["fallback"]
            source_type = result.get("source_type", "local")

            if fallback:
                logger.warning("Using fallback response (no stream returned)")
                for char in fallback:
                    yield f"data: {json.dumps({'type':'text','content':char})}\n\n"
                    time.sleep(0.005)

            elif response:
                logger.debug("Streaming response chunks to client")
                for chunk in response:
                    if chunk.text:
                        for char in chunk.text:
                            yield f"data: {json.dumps({'type':'text','content':char})}\n\n"
                            time.sleep(0.005)

        except Exception:
            logger.error("Exception during chat-stream generation", exc_info=True)
            if context:
                fallback = f"Based on the retrieved documents:\n\n{' '.join(c.get('text', '') for c in context[:5])}"
            else:
                fallback = "No relevant information found."
            for char in fallback:
                yield f"data: {json.dumps({'type':'text','content':char})}\n\n"
                time.sleep(0.005)

        sources = []
        seen = set()
        for c in context:
            key = (c["page"], c["source_file"])
            if key not in seen:
                seen.add(key)
                sources.append({"page": c["page"], "file": c["source_file"]})

        logger.info(f"Stream complete | sources_returned={len(sources)} | source_type={source_type}")
        yield f"data: {json.dumps({'type':'done','sources':sources,'retrieval_details':context,'source_type':source_type})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# NON-STREAMING (keep for fallback)
@app.post("/analyze-report")
async def analyze_report(file: UploadFile = File(...)):
    logger.info(f"POST /analyze-report | filename={file.filename} | content_type={file.content_type}")
    try:
        file_bytes = await file.read()
        logger.info(f"File received | size={len(file_bytes)} bytes")
        result = rag.analyze_report_pipeline(file_bytes, file.filename)
        if result.get("success"):
            logger.info(f"Report analysis complete | sources={len(result.get('sources', []))}")
        else:
            logger.warning(f"Report analysis failed | error={result.get('error')}")
        return result
    except Exception as e:
        logger.error(f"Unexpected error in /analyze-report | error={e}", exc_info=True)
        return {"success": False, "error": f"Server error: {str(e)}"}


#STEP 1 — extract report text + patient info, no analysis yet
@app.post("/extract-patient-info")
async def extract_patient_info_endpoint(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    logger.info(f"POST /extract-patient-info | user={current_user['username']} | filename={file.filename}")
    try:
        file_bytes = await file.read()
        extraction = rag.extract_report_text(file_bytes, file.filename)
        report_text = extraction["text"]
        extraction_method = extraction["method"]

        if not report_text:
            logger.warning("Could not extract text from uploaded report")
            return {"success": False, "error": "Could not read the report. Try a clearer image or text-based PDF."}

        patient_info = extract_patient_info(report_text, gemini_client=rag.gemini_client)
        logger.info(f"Patient info extracted | name={patient_info.name} | age={patient_info.age}")

        return {
            "success": True,
            "report_text": report_text,
            "extraction_method": extraction_method,
            "patient_info": patient_info.dict(),
        }
    except Exception as e:
        logger.error(f"Unexpected error in /extract-patient-info | error={e}", exc_info=True)
        return {"success": False, "error": f"Server error: {str(e)}"}


#STEP 2 request body: confirmed patient info + the already-extracted text
class AnalyzeRequest(BaseModel):
    report_text: str
    patient_info: PatientInfo
    extraction_method: Optional[str] = None
    filename: Optional[str] = None


# STREAMING REPORT ANALYSIS — now takes JSON (report_text + confirmed patient_info)
# instead of a re-uploaded file, since OCR already happened in /extract-patient-info
@app.post("/analyze-report-stream")
async def analyze_report_stream(req: AnalyzeRequest, current_user: dict = Depends(get_current_user)):
    logger.info(f"POST /analyze-report-stream | user={current_user['username']} | patient={req.patient_info.name} age={req.patient_info.age}")

    report_text = req.report_text
    extraction_method = req.extraction_method or "cached"
    patient_context = build_patient_context_block(req.patient_info)
    user_id = current_user["id"]
    filename = req.filename or "report"

    def generate():
        import time

        yield f"data: {json.dumps({'type':'status','content':'Identifying parameters...'})}\n\n"

        # Step 2: run TOA
        toa_result = rag.analyze_report_toa(report_text)

        # Send TOA data to frontend immediately
        yield f"data: {json.dumps({'type':'toa','thought': toa_result['thought'], 'observation': toa_result['observation'], 'search_terms': toa_result['search_terms']})}\n\n"

        yield f"data: {json.dumps({'type':'status','content':'Generating analysis...'})}\n\n"

        # Step 3: stream final analysis
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
        
        try:
            response = rag.gemini_client.models.generate_content_stream(
                model="gemini-2.5-flash-lite",
                contents=final_prompt
            )

            full_analysis_parts = []
            for chunk in response:
                if chunk.text:
                    full_analysis_parts.append(chunk.text)
                    for char in chunk.text:
                        yield f"data: {json.dumps({'type':'text','content':char})}\n\n"
                        time.sleep(0.003)

            full_analysis = "".join(full_analysis_parts)

        except Exception as e:
            logger.error(f"Streaming analysis failed: {e}", exc_info=True)
            yield f"data: {json.dumps({'type':'text','content':'Analysis generation failed. Please try again.'})}\n\n"
            full_analysis = ""

        # Send sources
        sources = []
        seen = set()
        for term, chunks in toa_result["action"].items():
            for chunk in chunks:
                key = (chunk.get("page"), chunk.get("source_file"))
                if key not in seen:
                    seen.add(key)
                    sources.append({"page": chunk.get("page"), "file": chunk.get("source_file"), "term": term})

        # Save this report to the DB, tied to the logged-in user only
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
        except Exception as e:
            logger.error(f"Failed to save report to database: {e}", exc_info=True)
            report_id = None

        yield f"data: {json.dumps({'type':'done','sources':sources,'extraction_method':extraction_method,'report_id':report_id})}\n\n"
        logger.info(f"Report stream complete | sources={len(sources)}")

    return StreamingResponse(generate(), media_type="text/event-stream")


#REPORT HISTORY — scoped to the logged-in user only

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

# MCP Integration
from fastapi_mcp import FastApiMCP

mcp = FastApiMCP(app)
mcp.mount()


@app.delete("/my-reports/{report_id}")
def delete_report(report_id: int, current_user: dict = Depends(get_current_user)):
    logger.info(f"DELETE /my-reports/{report_id} | user={current_user['username']}")
    report = database.get_report_by_id(report_id, current_user["id"])
    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")
    database.delete_report(report_id, current_user["id"])
    return {"success": True, "message": f"Report {report_id} deleted."}


@app.get("/my-reports/{report_id}/preview")
def preview_report(report_id: int, current_user: dict = Depends(get_current_user)):
    logger.info(f"GET /my-reports/{report_id}/preview | user={current_user['username']}")
    report = database.get_report_by_id(report_id, current_user["id"])
    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")
    return report
