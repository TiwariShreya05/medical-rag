"""
Guardrails module — scope enforcement for the medical assistant.

Ensures the RAG pipeline only answers medical / health-related queries.

Tiered check:
  1. Rule-based keyword match — free, instant, resolves the vast majority
     of real traffic.
  2. LLM classification call (via rag.generate_with_retry, which already
     has Gemini -> Groq fallback built in) — only used when the rule-based
     check is ambiguous.
"""

import logging
import re

logger = logging.getLogger("medical_rag")

OFF_TOPIC_MESSAGE = (
    "I'm a medical assistant and can only help with health and medical-related "
    "questions — symptoms, conditions, medications, reports, or finding doctors. "
    "Could you rephrase your question to be about a medical topic?"
)

_CLASSIFIER_PROMPT = """You are a strict scope classifier for a medical assistant application.

Decide if the user's query is related to medical, health, clinical, wellness, or healthcare topics — including symptoms, diagnoses, medications, treatments, anatomy, lab reports, mental health, nutrition, fitness/wellness advice, medical devices, insurance/medical admin, or finding doctors/clinics.

Also treat basic conversational messages directed at a medical assistant as IN scope (e.g. "hi", "thank you", "what can you help with").

Everything else (coding, general trivia, entertainment, politics, homework unrelated to health, etc.) is OUT of scope.

Query: "{query}"

Respond with exactly one word: MEDICAL or OFF_TOPIC"""

_GREETING_RE = re.compile(r"^\s*(hi|hello|hey|thanks|thank you)\s*[!.]?\s*$", re.I)

# Strong medical signal — if matched (and no off-topic signal), skip the LLM entirely
_MEDICAL_KEYWORDS = re.compile(
    r"\b("
    r"pain|ache|hurt|swelling|fever|cough|cold|flu|nausea|vomit|dizzy|fatigue|"
    r"symptom|diagnos|treatment|medicat|prescri|dose|dosage|surgery|therapy|"
    r"disease|infection|inflam|chronic|acute|"
    r"diabetes|hypertension|cholesterol|thyroid|cancer|tumou?r|asthma|copd|"
    r"heart|cardiac|kidney|liver|lung|brain|stomach|bone|joint|spine|skin|"
    r"blood pressure|blood sugar|glucose|hba1c|"
    r"doctor|physician|specialist|clinic|hospital|nurse|"
    r"report|scan|x-?ray|mri|ct scan|biopsy|lab (test|result)|"
    r"pregnan|period|menstru|mental health|anxiety|depression|stress|"
    r"allerg|rash|wound|injur|fracture|sprain|"
    r"vaccine|immun"
    r")\b",
    re.I,
)

# Strong non-medical signal — if matched (and no medical signal), skip the LLM entirely
_OFF_TOPIC_KEYWORDS = re.compile(
    r"\b("
    r"president|prime minister|vice president|election|parliament|"
    r"capital of|population of|"
    r"weather|stock price|cricket|football|movie|actor|song|"
    r"code|python|javascript|programming|debug|"
    r"recipe|cook(?!ie)|"
    r"who (is|was)|what is the capital|when did .* (win|happen)"
    r")\b",
    re.I,
)


def _rule_based_check(query: str) -> str | None:
    """Returns 'MEDICAL', 'OFF_TOPIC', or None (=> ambiguous, needs LLM)."""
    if _GREETING_RE.match(query):
        return "MEDICAL"

    has_medical = bool(_MEDICAL_KEYWORDS.search(query))
    has_off_topic = bool(_OFF_TOPIC_KEYWORDS.search(query))

    if has_medical and not has_off_topic:
        return "MEDICAL"
    if has_off_topic and not has_medical:
        return "OFF_TOPIC"
    return None  # ambiguous or no signal either way — let the LLM decide


def check_query_scope(query: str, rag_module) -> dict:
    """
    Returns {"in_scope": bool, "method": str, "raw_verdict": str | None}.

    `rag_module` is the already-imported rag module, passed in so this stays
    decoupled from rag.py's client setup and easy to unit test in isolation.

    Fails OPEN (allows the query through) if the classifier call errors out —
    an outage in the classifier shouldn't take down the whole assistant for
    real users. Failures are logged loudly so you notice if it's happening a lot.
    """
    query = (query or "").strip()
    if not query:
        return {"in_scope": True, "method": "empty", "raw_verdict": None}

    rule_verdict = _rule_based_check(query)
    if rule_verdict is not None:
        logger.info(
            f"Guardrail check | query={query!r} | verdict={rule_verdict!r} | method=rule_based"
        )
        return {"in_scope": rule_verdict == "MEDICAL", "method": "rule_based", "raw_verdict": rule_verdict}

    prompt = _CLASSIFIER_PROMPT.format(query=query)

    try:
        result_text = rag_module.generate_with_retry("gemini-2.5-flash", prompt)
        verdict = (result_text or "").strip().upper()
        in_scope = "MEDICAL" in verdict and "OFF_TOPIC" not in verdict
        logger.info(
            f"Guardrail check | query={query!r} | verdict={verdict!r} | method=classifier"
        )
        return {"in_scope": in_scope, "method": "classifier", "raw_verdict": verdict}
    except Exception as e:
        logger.error(f"Guardrail classifier failed, failing open: {e}", exc_info=True)
        return {"in_scope": True, "method": "fallback_open", "raw_verdict": None}