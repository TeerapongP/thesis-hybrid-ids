
from __future__ import annotations

import csv
import json
import re
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from dotenv import dotenv_values

from .langchain_client import MAX_REMEDIATION, invoke_structured

ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / ".env"
KB_PATH = ROOT / "data" / "knowledge_base" / "mitre_techniques.json"
MAPPING_PATH = ROOT / "config" / "mitre_mapping.csv"
MAPPING_DATASET = "cicids2017"
# Where the technique named in FACTS comes from. "rag" takes a 2-of-3 vote over the retrieved
# hits. "mapping" looks the RF-predicted label up in the MITRE table, which is not label
# leakage: it consumes the model's own prediction, downstream of the detection decision, the
# way a deployed IDS translates a classifier verdict into ATT&CK terms.
#
# "mapping" is deployed as of 2026-11-08, chosen on the measured ablation in CLAUDE.md
# ("Technique source ablation"), not on preference. It names a technique on 86 of 100 alerts
# against the vote's 50, agrees with the table on 99 of 100, and raises every diversity figure
# (45 distinct explanations against 41, echo 0.646 against 0.68). It lowers entity_pass from
# 0.96 to 0.838, and all 16 of those failures are the prose omitting a technique it was given,
# with no foreign or altered entity anywhere. Carrying a technique on 36 more alerts is
# 36 more chances to drop it. Read entity_pass beside entity_corruption_free, which stays 1.000.
#
# Kept as a named set rather than a boolean so a third source does not change signatures.
TechniqueSource = Literal["rag", "mapping"]
TECHNIQUE_SOURCES: tuple[TechniqueSource, ...] = ("rag", "mapping")
DEFAULT_TECHNIQUE_SOURCE: TechniqueSource = "mapping"
# What retrieval contributes to FACTS. "hits" pastes the top-3 retrieved chunks, whole, which is
# what the layer has always done and is still the deployed behaviour. "technique_kb" replaces
# them with the KB's detection analytics and mitigations for the one technique
# `resolve_technique` named, and nothing else from retrieval enters the prompt.
#
# Added to test one hypothesis: the chunks lose to the few-shot examples because they are the
# wrong shape, three whole technique documents handed to a model asked for two sentences and a
# short list. The default stays "hits" until a measurement says otherwise.
RetrievalContext = Literal["hits", "technique_kb"]
RETRIEVAL_CONTEXTS: tuple[RetrievalContext, ...] = ("hits", "technique_kb")
DEFAULT_RETRIEVAL_CONTEXT: RetrievalContext = "hits"
# Bounds on the KB text. Whole items only: an analytic line or a mitigation's first sentence is
# included complete or not at all, never cut at a character count.
MAX_DETECTION_ANALYTICS = 3
MAX_KB_MITIGATIONS = 3
MAX_MITIGATION_SUMMARY_CHARS = 240
REQUIRED_ENV_KEYS = (
    "LLM_BACKEND", "LLM_BASE_URL", "LLM_MODEL", "LLM_TIMEOUT_S", "LLM_MAX_TOKENS", "LLM_TEMPERATURE",
)
OUTPUT_FIELDS = (
    "priority", "explanation_th", "technique_id", "mitre_tactic", "affected_asset", "remediation",
    "confidence", "response_ms",
)
# Provisional, not tuned on data. The comment here used to justify it by `rf_ae_or`, a fusion
# rule retired on 2026-11-01; under the deployed cascade fused_score is RF's probability on
# RF-decided rows and `(if_norm + ae_norm) / 2` on the rest. Those are different scales, which
# is why the fallback branch needs its own cutoff rather than this one -- see
# PRIORITY_HIGH_MIN_SCORE_FALLBACK below and src/evaluation/derive_priority_cutoffs.py.
PRIORITY_HIGH_MIN_SCORE = 0.80
# The fallback branch's own HIGH cutoff, derived by src/evaluation/derive_priority_cutoffs.py
# as the p99.5 of that branch's train-BENIGN fused-score distribution -- "this far into the
# tail of normal traffic for this branch", the same form as every siem_baseline threshold.
#
# It is NOT derived by matching RF's 0.80, which two attempts showed to be vacuous: the 0.95
# confidence gate forces an RF-decided row to score >= 0.975 or <= 0.025, so 100% of
# RF-decided alerts already clear 0.80 and that branch can never emit a MEDIUM. Matching a
# vacuous threshold reproduced its vacuity -- the share-matched anchor landed on 0.0301, the
# alert threshold itself, which would page an analyst for all 720 fallback alerts including
# its 412 false positives.
#
# What p99.5 buys is small and must not be oversold: on the test split 6 of 720 fallback
# alerts become HIGH, 4 of them real attacks. That is the only swept value where paging beats
# a coin flip, and 304 of the 308 real attacks on this branch still do not page. The fallback
# score barely ranks attacks above benign within its own branch.
PRIORITY_HIGH_MIN_SCORE_FALLBACK = 0.4349
# The examples carry PLACEHOLDERS, never realistic values. Two earlier attempts kept the
# entities realistic (just changed which ones) and the model still copied them verbatim into
# real alerts: measured at 30/100 in reports/llm_faithfulness_raw.jsonl. A placeholder has
# nothing plausible to copy, and _assert_no_foreign_entities below rejects it if one escapes.
# Templates, never sent as-is. `render_few_shots` substitutes the live alert's own values
# before they reach the prompt.
#
# Why this shape. Realistic example values were copied into outputs (30/100 alerts, one a
# near-verbatim copy). Replacing them with {{...}} placeholders was meant to leave nothing
# plausible to copy; the model copied the placeholders themselves, failing the entity guard on
# 100% of alerts -- a total outage rather than a leak. Rendering per alert removes the generic
# token: a copied value is now the right value for that alert, and the example still
# demonstrates that entities must be named rather than omitted (omission was the dominant
# failure mode at 72 of 100 strict failures).
#
# The risk this trades into is template-echoing: an output that names the right host but
# reuses the example's sentence. That is measured, not assumed away -- see the
# `example_echo` metric in src/evaluation/eval_llm_faithfulness.py.
FEW_SHOT_TEMPLATES = [
    {
        "input": {
            "source_ip": "{{SRC_IP}}", "dst_ip": "{{DST_IP}}", "dst_port": "{{DST_PORT}}",
            "affected_asset": "{{ASSET}}", "rf_predicted_label": "{{LABEL}}",
            "technique_id": "{{TECHNIQUE_ID}}",
            "mitre_tactic": "{{TACTIC}}", "priority": "HIGH",
        },
        "output": {
            # The behaviour clause is deliberately class-neutral. It used to read "ส่งคำขอ
            # ผิดรูปแบบจำนวนมาก" -- many malformed requests -- which describes a flood. But
            # {{LABEL}} renders this alert's OWN predicted class into the example input, so
            # the prompt asserted that malformed-request flooding is the right explanation
            # for whatever class this alert is. On a PortScan the model copied it and
            # produced a confidently wrong sentence (seen on the first full-pipeline run:
            # label PortScan, technique T1046, explanation "malformed requests").
            #
            # Copying itself could not be stopped: three prompt variants were measured and
            # echo stayed at 60/100 (reports/llm_example_echo_experiment.*). So the fix is
            # to make the sentence that gets copied TRUE for any class rather than false for
            # most of them. This trades a correctness failure for a vagueness one, which is
            # the better of the two, and it is a limitation to report rather than a cure.
            # The same applies to the remediation lines: patching is specific to an
            # exploited service, so step 3 now asks for an assessment instead.
            "explanation_th": (
                "เครื่อง {{SRC_IP}} มีพฤติกรรมการเชื่อมต่อที่ผิดปกติไปยังบริการของ {{ASSET}} "
                "ที่พอร์ต {{DST_PORT}} ลักษณะตรงกับเทคนิค {{TECHNIQUE_ID}} "
                "จัดเป็นเหตุการณ์ระดับ {{PRIORITY}}"
            ),
            "remediation": [
                "ตรวจสอบ log ของ {{ASSET}} ในช่วงเวลาที่เกิดเหตุ",
                "ปิดกั้น {{SRC_IP}} ชั่วคราวที่อุปกรณ์ควบคุมการเข้าถึงที่เหมาะกับพอร์ต {{DST_PORT}}",
                "ประเมินความเสี่ยงของบริการบนพอร์ต {{DST_PORT}} และแก้ไขตามผลการประเมิน",
            ],
        },
    },
    {
        "input": {
            "source_ip": "{{SRC_IP}}", "dst_ip": "{{DST_IP}}", "dst_port": "{{DST_PORT}}",
            "affected_asset": "{{ASSET}}", "rf_predicted_label": "BENIGN",
            "technique_id": None, "mitre_tactic": None,
            "priority": "LOW",
        },
        "output": {
            # Same reason as shot 1: this clause used to assert the connection volume was
            # above normal, which a copied sentence would state as fact about a flow nobody
            # measured that way. What this shot must actually demonstrate is the null-
            # technique case -- name the entities, say the evidence was insufficient, keep
            # technique_id null -- so the volume claim is dropped and only that remains.
            "explanation_th": (
                "พบการเชื่อมต่อจาก {{SRC_IP}} ไปยัง {{ASSET}} ที่พอร์ต {{DST_PORT}} "
                "แต่หลักฐานจาก RAG ไม่สอดคล้องกันพอที่จะระบุเทคนิค MITRE ได้ "
                "จึงจัดเป็นเหตุการณ์ระดับ {{PRIORITY}}"
            ),
            "remediation": [
                "เฝ้าดูพฤติกรรมของ {{SRC_IP}} ต่อในรอบถัดไป",
                "เทียบปริมาณการเชื่อมต่อของ {{ASSET}} กับค่าปกติ",
            ],
        },
    },
]

_SHOT_FIELDS = ("SRC_IP", "DST_IP", "DST_PORT", "ASSET", "LABEL", "PRIORITY",
                "TECHNIQUE_ID", "TACTIC")


def _shot_substitutions(facts: dict[str, Any]) -> dict[str, str]:
    """Map each template placeholder to this alert's real value.

    `affected_asset` falls back to dst_ip because CICIDS-2017 carries no hostnames, and the
    asset under attack IS the destination. Anything still missing renders as a short Thai
    description rather than an empty string, so a copied example never reads as truncated.
    """
    ctx = facts.get("alert_context") or {}
    return {
        "SRC_IP": str(ctx.get("source_ip") or "ไม่ทราบต้นทาง"),
        "DST_IP": str(ctx.get("dst_ip") or "ไม่ทราบปลายทาง"),
        "DST_PORT": str(ctx.get("dst_port") if ctx.get("dst_port") is not None else "ไม่ทราบพอร์ต"),
        "ASSET": str(ctx.get("affected_asset") or ctx.get("dst_ip") or "ไม่ทราบระบบปลายทาง"),
        "LABEL": str(ctx.get("rf_predicted_label") or "ไม่ระบุ"),
        "PRIORITY": str(facts.get("priority") or "ไม่ระบุ"),
        "TECHNIQUE_ID": str(facts.get("technique_id") or "ไม่ระบุเทคนิค"),
        "TACTIC": str(facts.get("mitre_tactic") or "ไม่ระบุยุทธวิธี"),
    }


def _render_value(value: Any, subs: dict[str, str]) -> Any:
    if isinstance(value, str):
        for key, replacement in subs.items():
            value = value.replace("{{" + key + "}}", replacement)
        return value
    if isinstance(value, list):
        return [_render_value(v, subs) for v in value]
    if isinstance(value, dict):
        return {k: _render_value(v, subs) for k, v in value.items()}
    return value


def render_few_shots(facts: dict[str, Any]) -> list[dict[str, Any]]:
    """The few-shot examples with every placeholder replaced by this alert's own values."""
    subs = _shot_substitutions(facts)
    templates = FEW_SHOT_TEMPLATES
    if not facts.get("technique_id"):
        # Shot 1 demonstrates the known-technique case, so rendering it for an alert that
        # has no technique makes it assert one that does not exist. TECHNIQUE_ID then
        # resolves to the Thai stand-in "ไม่ระบุเทคนิค" and the clause reads
        # "ลักษณะตรงกับเทคนิค ไม่ระบุเทคนิค" -- broken Thai, which one alert in the
        # measured run copied verbatim. Shot 2 already covers the null case, so shot 1
        # keeps its structure here with the technique clause removed rather than being
        # dropped: dropping to a single example was measured as far worse (variant C,
        # echo 100/100).
        templates = json.loads(json.dumps(FEW_SHOT_TEMPLATES, ensure_ascii=False))
        shot = templates[0]
        shot["input"]["technique_id"] = None
        shot["input"]["mitre_tactic"] = None
        shot["output"]["explanation_th"] = (
            "เครื่อง {{SRC_IP}} มีพฤติกรรมการเชื่อมต่อที่ผิดปกติไปยังบริการของ {{ASSET}} "
            "ที่พอร์ต {{DST_PORT}} จัดเป็นเหตุการณ์ระดับ {{PRIORITY}}"
        )
    rendered = _render_value(templates, subs)
    leftover = [t for shot in rendered for t in _PLACEHOLDER_RE.findall(json.dumps(shot, ensure_ascii=False))]
    assert not leftover, f"few-shot templates still contain unrendered placeholders: {sorted(set(leftover))}"
    return rendered


# Anything the model emits that looks like one of these but is absent from FACTS is a leak.
_IPV4_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_TECHNIQUE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
_HOSTNAME_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+){1,}\b")
# Placeholders are example scaffolding. Emitting one verbatim is the failure mode the
# placeholder examples introduced: measured at 100/100 before this was added to the guard.
_PLACEHOLDER_RE = re.compile(r"\{\{[^}]*\}\}")


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
    foreign: list[str] = list(dict.fromkeys(_PLACEHOLDER_RE.findall(text)))
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


def derive_priority(fused_score: float, fused_prediction: int, model_used: str | None = None) -> str:
    """Tier an alert, using the cutoff belonging to the branch that produced the score.

    `model_used` defaults to the RF cutoff rather than the fallback's, which is the stricter
    of the two: a corrupted or missing branch label must not make the system page more
    often. That default also preserves the behaviour of any caller not yet migrated.
    """
    if fused_prediction != 1:
        return "LOW"
    cutoff = (PRIORITY_HIGH_MIN_SCORE_FALLBACK if model_used == "if_ae_fallback"
              else PRIORITY_HIGH_MIN_SCORE)
    return "HIGH" if fused_score >= cutoff else "MEDIUM"


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


@lru_cache(maxsize=None)
def _label_to_technique() -> dict[str, str]:
    with MAPPING_PATH.open(newline="", encoding="utf-8-sig") as fh:
        return {row["source_label"]: row["technique_id"] for row in csv.DictReader(fh)
                if row["dataset"] == MAPPING_DATASET and row["technique_id"]}


@lru_cache(maxsize=None)
def _kb_techniques() -> dict[str, Any]:
    return json.loads(KB_PATH.read_text(encoding="utf-8"))["techniques"]


def technique_from_label(predicted_label: str | None) -> tuple[str | None, str | None]:
    """The MITRE technique and tactic that config/mitre_mapping.csv gives a CICIDS-2017 label.

    Pass the label the MODEL predicted (`rf_predicted_label`), never the ground-truth label.
    A deployed IDS has the prediction and consuming it here sits downstream of the detection
    decision, so it is not leakage. Keying on the true label would hand the explanation layer
    the answer the detector is being measured on.

    The table names the technique and the KB supplies its properties. The tactic is read from
    the KB entry in the KB's own lowercase-hyphenated form, as `_technique_context` does, and
    not from the CSV's Title Case `tactic` column. That keeps every downstream `mitre_tactic`
    value unchanged, so only the technique's identity differs between the two sources.

    BENIGN and any label the table lacks return (None, None), which the prompt already
    treats as "evidence insufficient".
    """
    technique_id = _label_to_technique().get(predicted_label or "")
    if not technique_id:
        return None, None
    tactics = _kb_techniques().get(technique_id, {}).get("tactics") or []
    return technique_id, tactics[0] if tactics else None


def resolve_technique(
    technique_source: TechniqueSource,
    alert_context: dict[str, Any],
    rag_hits: list[dict[str, Any]],
) -> tuple[str | None, str | None]:
    if technique_source == "rag":
        return _technique_context(rag_hits)
    if technique_source == "mapping":
        return technique_from_label(alert_context.get("rf_predicted_label"))
    raise ValueError(f"unknown technique_source {technique_source!r}; expected one of {TECHNIQUE_SOURCES}")


_ANALYTIC_RE = re.compile(r"^-\s*Analytic\s+\d+:\s*(.+)$")
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _detection_analytics(detection: str) -> list[str]:
    """Up to MAX_DETECTION_ANALYTICS analytic lines, network telemetry first.

    The KB's detection field opens with a title line, which is dropped: it repeats the
    technique name and carries a corrupted dash from the source extraction. Analytics that
    read NSM (network security monitoring) logs come first because the alerts are flow records,
    so a Sysmon or macOS analytic is the least usable one to show. The sort is stable, so
    otherwise the KB's own order holds.
    """
    analytics = []
    for line in detection.splitlines():
        match = _ANALYTIC_RE.match(line.strip())
        if match:
            analytics.append(match.group(1).strip())
    analytics.sort(key=lambda text: "NSM:" not in text)
    return analytics[:MAX_DETECTION_ANALYTICS]


def _first_sentence(description: str) -> str | None:
    """A mitigation's opening sentence, or None when it is not short and clean enough to quote.

    None means the mitigation is shown by name alone. That is deliberate: a sentence cut at a
    character count reads as a claim the KB never made, which is worse than a bare name.
    """
    first = _SENTENCE_BREAK_RE.split(" ".join(description.split()), maxsplit=1)[0]
    if not first.endswith(".") or len(first) > MAX_MITIGATION_SUMMARY_CHARS or "�" in first:
        return None
    return first


def technique_reference(technique_id: str | None) -> dict[str, Any] | None:
    """The bounded detection and mitigation text the KB holds for one technique."""
    entry = _kb_techniques().get(technique_id or "")
    if not entry:
        return None
    mitigations = []
    for item in (entry.get("mitigations") or [])[:MAX_KB_MITIGATIONS]:
        shown = {"name": item["name"]}
        summary = _first_sentence(item.get("description") or "")
        if summary:
            shown["summary"] = summary
        mitigations.append(shown)
    detection = _detection_analytics(entry.get("detection") or "")
    if not detection and not mitigations:
        return None
    return {"detection": detection, "mitigations": mitigations}


def retrieval_facts(
    retrieval_context: RetrievalContext,
    technique_id: str | None,
    rag_hits: list[dict[str, Any]],
) -> dict[str, Any]:
    """The FACTS entry holding whatever retrieval contributes under the chosen context."""
    if retrieval_context == "hits":
        return {"rag_hits": rag_hits}
    if retrieval_context == "technique_kb":
        return {"technique_reference": technique_reference(technique_id)}
    raise ValueError(
        f"unknown retrieval_context {retrieval_context!r}; expected one of {RETRIEVAL_CONTEXTS}")


def build_prompt(
    alert_context: dict[str, Any],
    rag_hits: list[dict[str, Any]],
    fused_score: float,
    fused_prediction: int,
    confidence: float,
    technique_source: TechniqueSource = DEFAULT_TECHNIQUE_SOURCE,
    retrieval_context: RetrievalContext = DEFAULT_RETRIEVAL_CONTEXT,
) -> str:
    technique_id, tactic = resolve_technique(technique_source, alert_context, rag_hits)
    # `confidence` is |P(attack) - 0.5| * 2 -- certainty in EITHER direction, high both when
    # the model is sure it IS an attack and sure it is NOT. Shown to the LLM it reads as
    # certainty-of-attack and produced "ความมั่นใจต่ำ (0.99)" on benign rows in 12/100 alerts.
    # It is removed rather than renamed, because a renamed ambiguous number is still
    # ambiguous, and fused_score already carries risk in one direction. It is stripped from
    # the nested alert_context too, which carried a second copy.
    shown_context = {k: v for k, v in alert_context.items() if k != "confidence"}
    facts = {
        "alert_context": shown_context,
        "fused_score": fused_score,
        "fused_prediction": fused_prediction,
        "priority": derive_priority(fused_score, fused_prediction,
                                    alert_context.get("model_used")),
        "technique_id": technique_id,
        "mitre_tactic": tactic,
        **retrieval_facts(retrieval_context, technique_id, rag_hits),
    }
    return (
        "คุณเป็นนักวิเคราะห์ SOC ตอบเป็น JSON เท่านั้น ห้ามแต่ง IP, port, hostname, technique_id, "
        "mitre_tactic หรือ priority และต้องคัดลอกค่าจาก FACTS ตรงตัว "
        "ห้ามกล่าวถึงระดับความมั่นใจของโมเดลเป็นตัวเลข ให้สื่อความรุนแรงผ่าน priority แทน "
        "ความรุนแรงของคำอธิบายและ remediation ต้องสอดคล้องกับ priority ใน FACTS หาก technique_id เป็น null "
        "ให้คง null อธิบายว่าหลักฐาน RAG ไม่พอ และห้ามเอ่ยถึงรหัสเทคนิคใดๆ ในคำอธิบาย "
        "EXAMPLES แสดงรูปแบบคำตอบโดยใช้ค่าจริงของเหตุการณ์นี้แล้ว ให้เลียนรูปแบบ "
        "แต่เขียนเนื้อหาที่ตรงกับ FACTS ของเหตุการณ์นี้เอง ห้ามคัดลอกประโยคทั้งประโยค "
        f"remediation ต้องเป็นรายการข้อความภาษาไทย 1-{MAX_REMEDIATION} ข้อ ไม่ซ้ำกัน\n"
        f"SCHEMA={json.dumps(OUTPUT_FIELDS, ensure_ascii=False)}\n"
        # TWO examples, deliberately, after measuring one against two.
        #
        # One example scores BETTER on every faithfulness metric -- entity_pass 1.000,
        # entity_complete 1.000, hallucination 0.0000 -- and is the wrong choice. With a
        # single template to pattern-match, the model fills it in and nothing else: only
        # 33/100 explanations are distinct, and all 100 describe the flow as "ส่งคำขอ
        # ผิดรูปแบบจำนวนมาก" (sending many malformed requests), including all 10 BENIGN
        # flows, where it is simply false.
        #
        # Every metric scored that as perfect, because the entities are all correct and the
        # technique ids are all grounded. M2 checks whether the named entities match the
        # alert; it cannot check whether the described BEHAVIOUR matches the attack type.
        # Optimising against it would have selected the degenerate configuration.
        #
        # Two examples: 48/100 distinct, the phrase appears on 52/100 and on ZERO benign
        # flows. Measured in reports/llm_faithfulness_*.variant{B,C}* .
        f"EXAMPLES={json.dumps(render_few_shots(facts), ensure_ascii=False)}\n"
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
    technique_source: TechniqueSource = DEFAULT_TECHNIQUE_SOURCE,
    retrieval_context: RetrievalContext = DEFAULT_RETRIEVAL_CONTEXT,
) -> dict[str, Any]:
    technique_id, tactic = resolve_technique(technique_source, alert_context, rag_hits)
    prompt = build_prompt(alert_context, rag_hits, fused_score, fused_prediction, confidence,
                          technique_source, retrieval_context)
    guard_facts = {
        "alert_context": alert_context,
        "technique_id": technique_id,
        "mitre_tactic": tactic,
        **retrieval_facts(retrieval_context, technique_id, rag_hits),
    }
    started = time.perf_counter()
    parsed = parse_llm_response(call_llm(prompt, config))
    try:
        _assert_no_foreign_entities(parsed, guard_facts)
    except ValueError:
        # One corrective retry. The instruction alone was measured at 30% leakage, so a
        # single retry is a cheap second chance, not the control; the re-check below is.
        retry_prompt = prompt + (
            "\nคำตอบก่อนหน้าผิด เพราะมีชื่อเครื่อง ไอพี พอร์ต หรือรหัสเทคนิคที่ไม่ปรากฏใน FACTS "
            "ให้เขียนใหม่โดยใช้เฉพาะค่าที่อยู่ใน FACTS เท่านั้น เช่น source_ip dst_ip dst_port technique_id"
        )
        parsed = parse_llm_response(call_llm(retry_prompt, config))
        _assert_no_foreign_entities(parsed, guard_facts)
    response_ms = round((time.perf_counter() - started) * 1000)
    result = {
        "priority": derive_priority(fused_score, fused_prediction,
                                    alert_context.get("model_used")),
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
