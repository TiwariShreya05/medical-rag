"""
Patient info extraction module for Medical RAG Chatbot.

Save this as its OWN file: backend/patient_extraction.py (next to main.py).
Do NOT paste this into main.py — main.py just imports from it.

Strategy:
1. Try regex-based extraction first (fast, free, works for most printed lab reports).
2. For any field regex couldn't find, fall back to a single Gemini call that
   extracts ONLY the missing fields, returning strict JSON.
"""

import re
import json
import logging
from typing import Optional, List
from pydantic import BaseModel, Field

logger = logging.getLogger("medical_rag")


class PatientInfo(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None          # "Male" / "Female" / "Other"
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    blood_group: Optional[str] = None
    allergies: List[str] = Field(default_factory=list)


# ---------- Regex extraction ----------

NAME_PATTERNS = [
    r"(?:Patient\s*Name|Name)\s*[:\-]\s*([^\n\r]{2,40})",
]
AGE_PATTERNS = [
    r"Age\s*[:\-]\s*(\d{1,3})\s*(?:Y|Yrs|Years)?",
    r"(\d{1,3})\s*(?:Y|Yrs|Years)\s*/\s*(?:M|F|Male|Female)",  # "45 Y / M" style
]
SEX_PATTERNS = [
    r"(?:Sex|Gender)\s*[:\-]\s*(Male|Female|M|F|Other)\b",
    r"\d{1,3}\s*(?:Y|Yrs|Years)?\s*/\s*(M|F|Male|Female)\b",
]
HEIGHT_PATTERNS = [
    r"Height\s*[:\-]\s*(\d{2,3}(?:\.\d+)?)\s*cm",
]
WEIGHT_PATTERNS = [
    r"Weight\s*[:\-]\s*(\d{1,3}(?:\.\d+)?)\s*kg",
]
BLOOD_GROUP_PATTERNS = [
    r"Blood\s*Group\s*[:\-]\s*(A|B|AB|O)\s*[\+\-]?(?:ve|positive|negative)?",
]
ALLERGY_PATTERNS = [
    r"(?:Known\s*)?Allerg(?:y|ies)\s*[:\-]\s*([^\n]{2,150})",
]


def _first_match(patterns: List[str], text: str) -> Optional[str]:
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def _normalize_sex(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    raw = raw.strip().lower()
    if raw in ("m", "male"):
        return "Male"
    if raw in ("f", "female"):
        return "Female"
    return "Other"


def extract_patient_info_regex(text: str) -> PatientInfo:
    name = _first_match(NAME_PATTERNS, text)
    age_raw = _first_match(AGE_PATTERNS, text)
    sex_raw = _first_match(SEX_PATTERNS, text)
    height_raw = _first_match(HEIGHT_PATTERNS, text)
    weight_raw = _first_match(WEIGHT_PATTERNS, text)
    bg_raw = _first_match(BLOOD_GROUP_PATTERNS, text)
    allergy_raw = _first_match(ALLERGY_PATTERNS, text)

    allergies = []
    if allergy_raw:
        if allergy_raw.strip().lower() in ("none", "nil", "no known allergies", "n/a"):
            allergies = []
        else:
            allergies = [a.strip() for a in re.split(r",|;", allergy_raw) if a.strip()]

    return PatientInfo(
        name=name,
        age=int(age_raw) if age_raw and age_raw.isdigit() else None,
        sex=_normalize_sex(sex_raw),
        height_cm=float(height_raw) if height_raw else None,
        weight_kg=float(weight_raw) if weight_raw else None,
        blood_group=bg_raw.strip() if bg_raw else None,
        allergies=allergies,
    )


# ---------- Gemini fallback for fields regex missed ----------

def _missing_fields(info: PatientInfo) -> List[str]:
    missing = []
    if not info.name:
        missing.append("name")
    if not info.age:
        missing.append("age")
    if not info.sex:
        missing.append("sex")
    if not info.height_cm:
        missing.append("height_cm")
    if not info.weight_kg:
        missing.append("weight_kg")
    if not info.blood_group:
        missing.append("blood_group")
    if not info.allergies:
        missing.append("allergies")
    return missing


def extract_patient_info_llm(text: str, missing: List[str], gemini_client, model: str = "gemini-2.5-flash") -> dict:
    """
    gemini_client: pass rag.gemini_client (the same client your final analysis
    prompt already uses via rag.gemini_client.models.generate_content_stream).
    This uses the non-streaming counterpart since we just need one JSON blob back.
    """
    prompt = f"""Extract ONLY the following patient fields from the medical report
text below: {", ".join(missing)}.

Return STRICT JSON only, no markdown fences, no commentary. Use null for any
field you cannot find. Schema:
{{"name": string|null, "age": number|null, "sex": "Male"|"Female"|"Other"|null,
"height_cm": number|null, "weight_kg": number|null, "blood_group": string|null,
"allergies": [string]}}

Report text:
\"\"\"{text[:4000]}\"\"\"
"""
    try:
        response = gemini_client.models.generate_content(model=model, contents=prompt)
        raw = response.text.strip()
        raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"LLM patient-info fallback failed: {e}")
        return {}


def extract_patient_info(text: str, gemini_client=None) -> PatientInfo:
    """Main entry point. Call this right after OCR produces report text."""
    info = extract_patient_info_regex(text)
    missing = _missing_fields(info)

    if missing and gemini_client is not None:
        llm_data = extract_patient_info_llm(text, missing, gemini_client)
        for field in missing:
            value = llm_data.get(field)
            if value:
                setattr(info, field, value)

    return info


def build_patient_context_block(info: PatientInfo) -> str:
    """
    Call this with the user-CONFIRMED PatientInfo (after frontend edits) and
    prepend the result to your final analysis prompt.
    """
    allergy_str = ", ".join(info.allergies) if info.allergies else "None reported"
    return f"""Patient Profile:
- Name: {info.name or "Not provided"}
- Age: {info.age or "Unknown"}
- Sex: {info.sex or "Unknown"}
- Height: {info.height_cm or "Unknown"} cm
- Weight: {info.weight_kg or "Unknown"} kg
- Blood Group: {info.blood_group or "Unknown"}
- Known Allergies: {allergy_str}

IMPORTANT: Use age- and sex-appropriate reference ranges when interpreting lab
values. Do NOT give identical recommendations to patients of different ages —
explicitly call out pediatric vs adult vs geriatric differences, dosing
adjustments, and flag any medication class that would be contraindicated given
the patient's allergies or profile above.
"""