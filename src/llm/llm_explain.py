
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from .langchain_client import invoke_structured

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
FEW_SHOTS = [
    {
        "input": {"source_ip": "10.20.14.87", "dst_port": 445, "technique_id": "T1046"},
        "output": {"priority": "HIGH", "explanation_th": "ตรวจพบ 10.20.14.87 ที่พอร์ต 445 สอดคล้องกับ T1046", "remediation": ["ตรวจสอบต้นทาง 10.20.14.87"]},
    },
    {
        "input": {"source_ip": "203.0.113.44", "dst_port": 22, "technique_id": "T1110.001"},
        "output": {"priority": "MEDIUM", "explanation_th": "ตรวจพบ 203.0.113.44 ที่พอร์ต 22 สอดคล้องกับ T1110.001", "remediation": ["ตรวจสอบบัญชีที่เกี่ยวข้อง"]},
    },
]


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
        "technique_id": technique_id,
        "mitre_tactic": tactic,
        "rag_hits": rag_hits,
    }
    return (
        "คุณเป็นนักวิเคราะห์ SOC ตอบเป็น JSON เท่านั้น ห้ามแต่ง IP, port, hostname, technique_id, "
        "mitre_tactic หรือ confidence และต้องคัดลอกค่าจาก FACTS ตรงตัว หาก technique_id เป็น null "
        "ให้คง null และอธิบายว่าหลักฐาน RAG ไม่พอ remediation ต้องเป็นรายการข้อความภาษาไทย\n"
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
    if data.get("priority") not in ("HIGH", "MEDIUM", "LOW"):
        raise ValueError("LLM priority must be HIGH, MEDIUM, or LOW")
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
    started = time.perf_counter()
    parsed = parse_llm_response(call_llm(prompt, config))
    response_ms = round((time.perf_counter() - started) * 1000)
    result = {
        "priority": parsed["priority"],
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
