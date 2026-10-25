
from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from .langchain_client import MAX_REMEDIATION, invoke_structured

ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / ".env"
KB_PATH = ROOT / "data" / "knowledge_base" / "mitre_techniques.json"
REQUIRED_ENV_KEYS = (
    "LLM_BACKEND", "LLM_BASE_URL", "LLM_MODEL", "LLM_TIMEOUT_S", "LLM_MAX_TOKENS", "LLM_TEMPERATURE",
)
OUTPUT_FIELDS = (
    "priority", "explanation_th", "technique_id", "mitre_tactic", "affected_asset", "remediation",
    "confidence", "response_ms",
)
# Provisional, not tuned on data. fused_score already folds RF and AE into one risk measure under
# rf_ae_or; rf confidence is not used because it is RF certainty in either direction.
PRIORITY_HIGH_MIN_SCORE = 0.80
# The examples carry PLACEHOLDERS, never realistic values. Two earlier attempts kept the
# entities realistic (just changed which ones) and the model still copied them verbatim into
# real alerts: measured at 30/100 in reports/llm_faithfulness_raw.jsonl. A placeholder has
# nothing plausible to copy, and _assert_no_foreign_entities below rejects it if one escapes.
FEW_SHOTS = [
    {
        "input": {
            "source_ip": "{{SRC_IP}}", "dst_ip": "{{DST_IP}}", "dst_port": "{{DST_PORT}}",
            "affected_asset": "{{ASSET}}", "rf_predicted_label": "{{LABEL}}",
            "confidence": "{{CONFIDENCE}}", "technique_id": "{{TECHNIQUE_ID}}",
            "mitre_tactic": "{{TACTIC}}", "priority": "HIGH",
        },
        "output": {
            "explanation_th": (
                "เครื่อง {{SRC_IP}} ส่งคำขอผิดรูปแบบจำนวนมากไปยังบริการของ {{ASSET}} "
                "ที่พอร์ต {{DST_PORT}} ลักษณะตรงกับเทคนิค {{TECHNIQUE_ID}} "
                "โมเดลมีความมั่นใจ {{CONFIDENCE}}"
            ),
            "remediation": [
                "ตรวจสอบ log ของ {{ASSET}} ในช่วงเวลาที่เกิดเหตุ",
                "ปิดกั้น {{SRC_IP}} ชั่วคราวที่อุปกรณ์ควบคุมการเข้าถึงที่เหมาะกับพอร์ต {{DST_PORT}}",
                "ตรวจสอบและติดตั้งแพตช์ของบริการบนพอร์ต {{DST_PORT}}",
            ],
        },
    },
    {
        "input": {
            "source_ip": "{{SRC_IP}}", "dst_ip": "{{DST_IP}}", "dst_port": "{{DST_PORT}}",
            "affected_asset": "{{ASSET}}", "rf_predicted_label": "BENIGN",
            "confidence": "{{CONFIDENCE}}", "technique_id": None, "mitre_tactic": None,
            "priority": "LOW",
        },
        "output": {
            "explanation_th": (
                "พบปริมาณการเชื่อมต่อจาก {{SRC_IP}} ไปยัง {{ASSET}} ที่พอร์ต {{DST_PORT}} สูงกว่าปกติ "
                "แต่หลักฐานจาก RAG ไม่สอดคล้องกันพอที่จะระบุเทคนิค MITRE ได้ "
                "และความมั่นใจต่ำ ({{CONFIDENCE}})"
            ),
            "remediation": [
                "เฝ้าดูพฤติกรรมของ {{SRC_IP}} ต่อในรอบถัดไป",
                "เทียบปริมาณการเชื่อมต่อของ {{ASSET}} กับค่าปกติ",
            ],
        },
    },
]

# Anything the model emits that looks like one of these but is absent from FACTS is a leak.
_IPV4_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_TECHNIQUE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
_HOSTNAME_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+){1,}\b")


def _facts_entity_pool(facts: dict[str, Any]) -> set[str]:
    """Every string a faithful output is allowed to contain an entity from."""
    pool: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif value is not None:
            pool.add(str(value))

    walk(facts)
    return pool


def find_foreign_entities(text: str, facts: dict[str, Any]) -> list[str]:
    """Entity-shaped tokens in `text` that appear nowhere in `facts`.

    Shape-based rather than a blocklist of the current example values, so the check keeps
    working when the examples change. Thai prose contains no ASCII hostnames or dotted
    quads, so this does not fire on ordinary wording.
    """
    pool = _facts_entity_pool(facts)
    joined = " ".join(pool)
    foreign: list[str] = []
    for pattern in (_IPV4_RE, _TECHNIQUE_RE, _HOSTNAME_RE):
        for token in pattern.findall(text):
            if token in pool or token in joined:
                continue
            if token not in foreign:
                foreign.append(token)
    return foreign


def _assert_no_foreign_entities(parsed: dict[str, Any], facts: dict[str, Any]) -> None:
    """Raise rather than return an explanation naming a host the alert never mentioned.

    Failing loudly is the point. A silently wrong hostname sends an analyst to the wrong
    machine, which is worse than no explanation at all.
    """
    text = " ".join([parsed["explanation_th"], *parsed["remediation"]])
    foreign = find_foreign_entities(text, facts)
    if foreign:
        raise ValueError(f"LLM output names entities absent from FACTS: {foreign}")


def load_env_config(env_path: Path = ENV_PATH) -> dict[str, Any]:
    raw = dotenv_values(env_path)
    missing = [key for key in REQUIRED_ENV_KEYS if not raw.get(key)]
    if missing:
        raise RuntimeError(f".env is missing required LLM keys: {missing}")
    backend = raw["LLM_BACKEND"].lower()
    if backend not in ("ollama", "llamacpp"):
        raise ValueError("LLM_BACKEND must be 'ollama' or 'llamacpp'")
    return {
        "backend": backend,
        "base_url": raw["LLM_BASE_URL"].rstrip("/"),
        "model": raw["LLM_MODEL"],
        "timeout_s": float(raw["LLM_TIMEOUT_S"]),
        "max_tokens": int(raw["LLM_MAX_TOKENS"]),
        "temperature": float(raw["LLM_TEMPERATURE"]),
    }


def derive_priority(fused_score: float, fused_prediction: int) -> str:
    if fused_prediction != 1:
        return "LOW"
    return "HIGH" if fused_score >= PRIORITY_HIGH_MIN_SCORE else "MEDIUM"


def _technique_context(rag_hits: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    technique_ids = [hit.get("metadata", {}).get("technique_id") for hit in rag_hits[:3]]
    technique_ids = [value for value in technique_ids if value]
    if not technique_ids:
        return None, None
    technique_id, votes = Counter(technique_ids).most_common(1)[0]
    if votes < 2:
        return None, None
    kb = json.loads(KB_PATH.read_text(encoding="utf-8"))["techniques"].get(technique_id, {})
    tactics = kb.get("tactics") or []
    return technique_id, tactics[0] if tactics else None


def build_prompt(
    alert_context: dict[str, Any],
    rag_hits: list[dict[str, Any]],
    fused_score: float,
    fused_prediction: int,
    confidence: float,
) -> str:
    technique_id, tactic = _technique_context(rag_hits)
    facts = {
        "alert_context": alert_context,
        "fused_score": fused_score,
        "fused_prediction": fused_prediction,
        "confidence": confidence,
        "priority": derive_priority(fused_score, fused_prediction),
        "technique_id": technique_id,
        "mitre_tactic": tactic,
        "rag_hits": rag_hits,
    }
    return (
        "คุณเป็นนักวิเคราะห์ SOC ตอบเป็น JSON เท่านั้น ห้ามแต่ง IP, port, hostname, technique_id, "
        "mitre_tactic, priority หรือ confidence และต้องคัดลอกค่าจาก FACTS ตรงตัว "
        "ความรุนแรงของคำอธิบายและ remediation ต้องสอดคล้องกับ priority ใน FACTS หาก technique_id เป็น null "
        "ให้คง null อธิบายว่าหลักฐาน RAG ไม่พอ และห้ามเอ่ยถึงรหัสเทคนิคใดๆ ในคำอธิบาย "
        "EXAMPLES ใช้ค่าตัวยึดในรูปแบบ {{...}} เท่านั้น ห้ามคัดลอกค่าใดจาก EXAMPLES "
        "และห้ามเขียนตัวยึด {{...}} ลงในคำตอบ ให้แทนด้วยค่าจริงจาก FACTS เสมอ "
        f"remediation ต้องเป็นรายการข้อความภาษาไทย 1-{MAX_REMEDIATION} ข้อ ไม่ซ้ำกัน\n"
        f"SCHEMA={json.dumps(OUTPUT_FIELDS, ensure_ascii=False)}\n"
        f"EXAMPLES={json.dumps(FEW_SHOTS, ensure_ascii=False)}\n"
        f"FACTS={json.dumps(facts, ensure_ascii=False, default=str)}"
    )


def call_llm(prompt: str, config: dict[str, Any] | None = None) -> str:
    config = config or load_env_config()
    return json.dumps(invoke_structured(prompt, config), ensure_ascii=False)


def parse_llm_response(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    data = json.loads(text)
    if not isinstance(data.get("remediation"), list) or not all(isinstance(x, str) for x in data["remediation"]):
        raise ValueError("LLM remediation must be a list of strings")
    if not isinstance(data.get("explanation_th"), str):
        raise ValueError("LLM explanation_th must be a string")
    return data


def explain_alert(
    alert_context: dict[str, Any],
    rag_hits: list[dict[str, Any]],
    fused_score: float,
    fused_prediction: int,
    confidence: float,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    technique_id, tactic = _technique_context(rag_hits)
    prompt = build_prompt(alert_context, rag_hits, fused_score, fused_prediction, confidence)
    guard_facts = {
        "alert_context": alert_context,
        "technique_id": technique_id,
        "mitre_tactic": tactic,
        "rag_hits": rag_hits,
    }
    started = time.perf_counter()
    parsed = parse_llm_response(call_llm(prompt, config))
    try:
        _assert_no_foreign_entities(parsed, guard_facts)
    except ValueError:
        # One corrective retry. The instruction alone was measured at 30% leakage, so a
        # single retry is a cheap second chance, not the control; the re-check below is.
        retry_prompt = prompt + (
            "\nคำตอบก่อนหน้ามีค่าที่ไม่ได้อยู่ใน FACTS ให้เขียนใหม่โดยใช้เฉพาะค่าจาก FACTS"
        )
        parsed = parse_llm_response(call_llm(retry_prompt, config))
        _assert_no_foreign_entities(parsed, guard_facts)
    response_ms = round((time.perf_counter() - started) * 1000)
    result = {
        "priority": derive_priority(fused_score, fused_prediction),
        "explanation_th": parsed["explanation_th"],
        "technique_id": technique_id,
        "mitre_tactic": tactic,
        "affected_asset": alert_context.get("affected_asset") or alert_context.get("dst_ip"),
        "remediation": parsed["remediation"],
        "confidence": float(confidence),
        "response_ms": response_ms,
    }
    assert tuple(result) == OUTPUT_FIELDS
    return result
