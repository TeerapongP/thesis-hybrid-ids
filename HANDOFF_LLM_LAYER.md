# HANDOFF: LLM Explanation Layer (Typhoon2.1 12B Instruct + RAG + Fusion → Thai JSON)

## Scope

> **Implementation update (2026-09-24):** `src/llm/llm_explain.py`,
> `src/evaluation/eval_llm_layer.py`, and `models/llm_layer_config.json` now implement the HTTP,
> constrained-JSON, schema, confidence, and offline contract paths. Confidence is RF decisiveness
> emitted by `fusion_layer.fuse_with_metadata`. Live Typhoon quality/throughput verification is
> still pending because Ollama is not installed on this machine.

This phase builds the **LLM explanation layer** described in `CLAUDE.md`'s "Project overview" §4: given the ML ensemble's fusion output (a risk score/prediction) plus alert context plus the RAG layer's retrieved MITRE ATT&CK context, produce a structured JSON explanation **in Thai**. Concretely: a prompt-construction + local-LLM-inference module, a reconciled JSON output schema (see Conventions below — the source diagram and the user's requested field set do not match 1:1 and this brief resolves that explicitly, it is not left to the developer to pick), and an evaluation harness for entity-preservation and faithfulness/hallucination behavior (architecture diagram tag `M2`).

**Model decision — already made, not an open question.** `CLAUDE.md` flags that the diagrams disagree on which local LLM to use (`diagrams/thesis_ml_llm_pipeline_v3.drawio`'s `llm` node names *"Qwen3.5-35B-A3B (LM Studio · Mac Mini M4 Pro 48 GB · local REST API)"*; `CLAUDE.md`'s own prose separately mentions a Llama 3.2 3B + DeepSeek-R1 7B combination from another diagram revision) and says to check with the user before assuming one. **The user has now explicitly resolved this: the model is Typhoon2.1 (12B) Instruct, served locally via Ollama or llama.cpp.** This supersedes both diagram-era choices — Qwen3.5-35B-A3B and the Llama 3.2 3B/DeepSeek-R1 7B pairing are dead options for this repo going forward. Do not re-litigate this; if a future diagram revision or teammate suggests otherwise, point them at this paragraph and this session's decision, not the (now-stale) diagrams.

**This phase explicitly IS:**
- A local-inference setup path for Typhoon2.1 12B Instruct (Ollama or llama.cpp), verified against this machine's actual hardware, not assumed to "just work."
- A structured Thai prompt template with entity-preservation few-shot examples.
- A thin, importable inference module (mirroring `src/modeling/fusion_layer.py`/`src/rag/rag_retrieval.py`'s pattern) that wires together `src/modeling/fusion_layer.py`'s fused score, `src/rag/rag_retrieval.py`'s `RagRetriever.retrieve()` output, and the LLM call into one structured-JSON-out function.
- A reconciled, explicit output JSON schema (see Conventions §"Output schema reconciliation").
- A concrete, named eval suite for `M2` (entity preservation + faithfulness + adversarial/known-gap behavior on Bot/Heartbleed).

**This phase explicitly is NOT:**
- Not a retrain or re-tuning of any ML ensemble member or the Fusion Layer. `models/random_forest.pkl`, `models/isolation_forest_behavioral_v2.pkl`, `models/autoencoder_behavioral_v2.pt`, `src/modeling/fusion_layer.py`'s `CONFIDENCE_THRESHOLD=0.95`/`FUSED_DECISION_THRESHOLD=0.05`/normalization constants are **frozen inputs** to this phase — read them, do not change them.
- Not a re-chunking or re-embedding of the RAG knowledge base. `src/rag/rag_retrieval.py`'s 45-chunk collection, its BM25 tokenizer, and its `dense_weight=0.5`/`bm25_weight=0.5` hybrid weighting are frozen inputs. If retrieval quality on some category looks bad to you, that is very likely the already-investigated, already-documented Bot/Heartbleed gap (see Required Reading #4) — do not "fix" it by changing retrieval code in this phase; that would be redesigning a different, already-closed phase.
- Not `enrich.py` (live OpenCTI GraphQL feed) — still unbuilt, out of scope.
- Not `alert_dispatcher.py` (priority-based routing/notification) — this phase produces the `priority` field the dispatcher will eventually consume, but does not build the dispatcher itself.
- Not the A/B analyst triage-time study (`diagrams/thesis_ml_llm_pipeline_v3.drawio`'s §⑤, tag `M3`/`M4`) — that is a separate, later human-subjects phase that *consumes* this layer's output; do not attempt to simulate or pre-empt it here.
- Not SHAP-based explainability — `CLAUDE.md` and `src/modeling/fusion_layer.py`'s own docstring both note SHAP integration remains unbuilt in the Fusion Layer; this phase's `explanation_th` is an LLM-generated narrative, not a SHAP-value-grounded one. Do not imply SHAP grounding in the prompt or output that isn't actually there.

## Required reading, in order

1. **`CLAUDE.md`** (repo root) — authoritative conventions, current pipeline state, the Anaconda-vs-`.venv` dependency split (system Anaconda Python — `C:\Users\devMrX\anaconda3\python.exe` — runs every real pipeline script; the checked-in `.venv` is `ipykernel`-only), and the GPU note under `03_autoencoder.ipynb`: this machine has an **RTX 3070** (a CUDA setup was done specifically for this card: `torch==2.12.1+cu130`). That GPU's VRAM budget is directly relevant to whether Typhoon2.1 12B fits — see Conventions §"Local inference feasibility" below, this is not something to assume works.
2. **`HANDOFF_RAG_PHASE3.md`** (repo root) — the most recent handoff of this exact shape/rigor in this repo; mirror its structure and its "verify before you write" discipline. Also documents the RAG layer's own scope boundaries, several of which this phase must respect (frozen chunking, no re-tuning against the eval set).
3. **`reports/rag_layer_report.txt`** §6/§6.1 and **`reports/rag_retrieval_precision_at_3.csv`** — the actual, current RAG retrieval quality per category, including the **Bot and Heartbleed permanent gaps** you must design adversarial test cases against (see Deliverables §3 below). Current aggregate: mean precision@3 = 0.5000, hit@3 = 12/14. Per the CSV, Bot's and Heartbleed's *actual* top-3 retrieved chunks are:
   - **Bot** (expected `T1071.001`, "Application Layer Protocol: Web Protocols"): top-3 retrieved = `T1046::detection` (technique_id `T1046`), `T1499.002::detection` (`T1499.002`), `T1498::detection` (`T1498`) — **zero of the top-3 match the expected technique**, `n_relevant_in_top3=0`.
   - **Heartbleed** (expected `T1190`, "Exploit Public-Facing Application (CVE-2014-0160)"): top-3 retrieved = `T1071.001::description` (`T1071.001`), `T1499.002::detection` (`T1499.002`), `T1110::description` (`T1110`) — again **zero of the top-3 match**, `n_relevant_in_top3=0`.
   These are read directly from `reports/rag_retrieval_precision_at_3.csv`'s Bot/Heartbleed rows — do not re-derive or approximate them, use these exact chunk ids/technique_ids in your adversarial test fixtures.
4. **`src/rag/rag_retrieval.py`** — read `RagRetriever.retrieve()` in full (lines ~322–371). Its actual return shape, per hit dict: `{"rank": int, "id": str (e.g. "T1046::detection" or "mitigation::M1037"), "document": str (the raw chunk text), "metadata": {"technique_id": str, "chunk_type": "description"|"detection"|"mitigation", "technique_name"?: str, "technique_ids"?: str (comma-separated, mitigation chunks only), "mitigation_id"?: str, "mitigation_name"?: str}, "combined_score": float, "dense_score": float, "bm25_score": float}`. Construct the module with `RagRetriever(config=load_env_config())` (loads/builds the persisted `data/chroma_db/` collection); calling `.retrieve(query, top_k=3)` is the only call this phase's prompt-builder needs.
5. **`src/modeling/fusion_layer.py`** + **`models/fusion_config.json`** — read the module docstring (Thai) and `fuse()` (lines ~94–120) in full. Actual signature: `fuse(rf_attack_probability, if_anomaly_score, ae_reconstruction_error) -> (fused_score: np.ndarray in [0,1], fused_prediction: np.ndarray of int 0/1)`. There is **no separate "confidence" value returned** — see Conventions §"Output schema reconciliation" for how this phase must derive one. Also read `models/fusion_config.json`'s `known_limitations` array verbatim — Bot near-zero recall and the PortScan LOAO gap are documented, root-caused ML-layer limitations that this phase's prompts/tests must not silently paper over (a confidently-worded Thai explanation must not overstate certainty the upstream model doesn't have).
6. **`diagrams/thesis_ml_llm_pipeline_v3.drawio`** — grep for `id="ml_out"`, `id="llm"`, `id="llm_out"`, `id="tag_m2"`. Exact node text (verified 2026-07-14):
   - `ml_out`: *"ML output — risk score (0–100) + alert context + explainability … risk_score · anomaly_type · top_3_features (SHAP) · source_ip · dst_port · timestamp · model_used · ensemble_confidence"*
   - `llm`: names Qwen3.5-35B-A3B — **superseded, see Scope above**.
   - `llm_out`: *"LLM structured output (JSON · ภาษาไทย) … priority: HIGH/MEDIUM/LOW · explanation_th: คำอธิบาย · mitre_tactic: Txxxx · affected_asset: hostname … remediation: [step1, step2, step3] · faithfulness_score · hallucination_rate · response_ms: ~2400"*
   - `tag_m2`: *"M2 — Faithfulness score / Hallucination rate"*
   Note the diagram's own internal inconsistency you must not carry forward uncorrected: it labels a field `mitre_tactic` but formats it as `Txxxx` — that is a **technique ID** format, not a tactic. Real MITRE tactics (see `data/knowledge_base/mitre_techniques.json`'s `tactics` field, e.g. `T1046`'s is `["discovery"]`, `T1110.001`'s is `["credential-access"]`) are lowercase kebab-case strings, never `Txxxx`. See Conventions §"Output schema reconciliation" for how this is resolved.
7. **`data/knowledge_base/mitre_techniques.json`** — the actual source of `technique_id`/`tactics`/`description`/`detection` text your prompts will quote from (via RAG retrieval, not directly). Confirms the `tactics` field format above.
8. **`config/mitre_mapping.csv`** — ground truth `(dataset, source_label, technique_id, technique_name, tactic, mapping_confidence)` per category. This is your **eval-only** query set (mirrors `src/evaluation/eval_rag_precision.py`'s own no-leakage rule) — never feed `technique_name`/`tactic` from this file into a live production prompt (that's the RAG retriever's/Fusion Layer's job to surface at inference time from independently-available signals), only use it to check eval-script assertions.
9. **`src/evaluation/eval_rag_precision.py`** — the precedent for this phase's eval script: same repo, same "read a CSV ground-truth query set, call the module under test, write per-category results to `reports/*.csv`, no blended single number" shape. Mirror its structure for `src/evaluation/eval_llm_layer.py` (see Deliverables).
10. **Root `.env`** — confirm what's already there before adding new keys (do not print its contents; secrets like `OPENCTI_TOKEN` live in it). It currently has zero LLM-related keys (`OPENCTI_*`, `KAFKA_*`, `CHROMA_*`/`EMBED_MODEL`/`RAG_*`, `LOG_LEVEL` only) — you will need to add new ones (see Conventions).

## Conventions that must be followed

### Local inference feasibility (Typhoon2.1 12B on this machine) — verify, do not assume
- This machine's GPU is an **RTX 3070** (confirmed via `CLAUDE.md`'s CUDA setup note for `03_autoencoder.ipynb`, which specifically matched a `torch` CUDA build to this card's driver). The RTX 3070 has **8GB VRAM**. A 12B-parameter model at 16-bit would need ~24GB just for weights — nowhere close to fitting. At 4-bit quantization (e.g. GGUF `Q4_K_M`, ~4.5–5 bits/weight average) a 12B model's weights alone are roughly 7–8GB, **before** KV-cache (grows with context length — your prompt will include the alert context, the RAG-retrieved chunk text, and few-shot examples, easily 1–2k+ tokens) and before Windows' own VRAM overhead. **This is genuinely tight, possibly not fully GPU-resident at full context length — do not assert it fits.** Concretely required of the developer before writing any prompt code:
  - Actually install Typhoon2.1 12B Instruct (via `ollama pull <verified-tag>` or a llama.cpp GGUF, whichever backend is chosen) and run a real inference call while watching `nvidia-smi` (or `ollama ps`) for actual VRAM usage and tokens/sec, at a prompt length representative of this phase's real prompts (see the few-shot template below — build one full real prompt and measure with *that*, not a toy one-line prompt).
  - If it does not fit fully on GPU: both Ollama and llama.cpp support **partial GPU offload** (Ollama: `num_gpu` / `OLLAMA_NUM_GPU`; llama.cpp: `-ngl <N>` layers-on-GPU flag) — a hybrid CPU+GPU split is a legitimate fallback, just slower. Record whichever configuration was actually used (full offload vs. partial, and the measured tokens/sec) in the deliverable config file (see below) the same way `models/autoencoder_config.json` records the CUDA/GPU choice made for that notebook — don't leave this undocumented.
  - Do not assume a specific Ollama model tag exists for "Typhoon2.1 12B Instruct" without checking (e.g. `ollama pull <tag>` actually succeeding, or a HuggingFace GGUF search actually turning up a working conversion) — this repo has zero prior reference to this model, unlike Qwen/Llama/DeepSeek which at least appeared in a diagram. If no ready GGUF/Ollama-library entry exists and one has to be converted from a HuggingFace safetensors release, that conversion step and its outcome must be recorded, not silently skipped.
- **Backend choice (Ollama vs. llama.cpp) is the developer's call**, but whichever is chosen must run as its own local server process (Ollama's default `ollama serve` on `localhost:11434`, or llama.cpp's `llama-server` binary) that the Python inference module calls over HTTP — **not** a from-source CUDA-compiled Python binding as the primary path (e.g. avoid `llama-cpp-python` built with `CMAKE_ARGS="-DGGML_CUDA=on"` unless you've confirmed that toolchain works on this machine; this repo's only confirmed working CUDA-linked Python package is `torch==2.12.1+cu130`, and there's no established precedent here for compiling llama.cpp's CUDA kernels from Python). A plain HTTP client (`requests`, or the `ollama` pip package) is lower-risk and mirrors this repo's pattern of small, inspectable code over heavyweight/fragile dependencies (`src/rag/rag_retrieval.py`'s own docstring makes exactly this argument for why it avoided LangChain).
- **Dependency placement**: whatever Python HTTP client library you use belongs on the **system Anaconda Python**, matching the precedent set for `chromadb`/`rank_bm25`/`sentence-transformers` in the RAG phase (see `reports/rag_layer_report.txt` §5) — confirmed there by checking `.venv/Lib/site-packages` first (found `ipykernel`-only) before installing anything on Anaconda. Do the same check-then-install discipline here; do not add these to `pyproject.toml`/`.venv` without first confirming that's actually wanted.

### Config-driven, not hardcoded
Add new `.env` keys for this phase (there are currently zero LLM-related keys in `.env` — verified), following the exact pattern the RAG phase used (6 new keys: `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_COLLECTION`, `EMBED_MODEL`, `RAG_TOP_K`, `RAG_REINDEX_HOURS`, all read via `src/rag/rag_retrieval.py`'s `load_env_config()`, which hard-fails if any are missing). Equivalent for this phase, at minimum: `LLM_BACKEND` (`ollama`|`llamacpp`), `LLM_BASE_URL` (e.g. `http://localhost:11434`), `LLM_MODEL` (the verified model/tag string), `LLM_TIMEOUT_S`, `LLM_MAX_TOKENS`, `LLM_TEMPERATURE`. Never hardcode these in code — load them the same defensive way (`load_env_config`-style, asserting presence) as `src/rag/rag_retrieval.py` does. Never print or commit `.env`'s contents.

### Output schema reconciliation — resolved here, do not re-decide
The diagram's `llm_out` field list and the user's requested field list (`technique_id`, `explanation_th`, `remediation`, `confidence`) do not match 1:1. Resolved as follows — implement exactly this, this is a decision, not a suggestion:

| Field | Diagram had | User asked for | Resolution |
|---|---|---|---|
| `priority` | ✓ (`HIGH`/`MEDIUM`/`LOW`) | — | **Keep.** Structurally required for the (unbuilt) `alert_dispatcher.py`'s documented HIGH/MEDIUM/LOW routing — dropping it would silently break a downstream contract `CLAUDE.md` already documents. |
| `explanation_th` | ✓ | ✓ | **Keep**, both agree. |
| `technique_id` | ✗ (diagram had no such field) | ✓ | **Add.** This is the real `Txxxx` value — sourced from the RAG retriever's top-1 hit's `metadata["technique_id"]` (or the Fusion Layer's/RF's own predicted category → `config/mitre_mapping.csv` lookup as a secondary signal — see Deliverables for exactly which one wins when they disagree, that logic must be explicit in code, not implicit). |
| `mitre_tactic` | ✓, but mislabeled — formatted as `Txxxx` in the diagram, which is actually a technique ID, not a tactic | — | **Keep the field name, fix its content.** `mitre_tactic` must hold the real MITRE tactic string (lowercase, e.g. `"discovery"`, `"credential-access"` — see `data/knowledge_base/mitre_techniques.json`'s `tactics` field), sourced from the same technique the `technique_id` field names. This is a deliberate correction of the diagram's own internal inconsistency (`Txxxx` is never a valid tactic value anywhere else in this repo — `config/mitre_mapping.csv`'s own `tactic` column also uses this lowercase form), not a silent reinterpretation — call this out in your `models/llm_layer_config.json` sidecar the same way `models/rag_retrieval_config.json` documents its own resolved ambiguities. |
| `affected_asset` | ✓ (hostname) | — | **Keep.** Populate from whatever hostname/asset identifier the alert context carries; if only `source_ip`/`dst_ip` is available (no resolved hostname — likely, since none of `feature_extractor.py`'s outputs carry a hostname column), fall back to the IP itself rather than fabricating a hostname. Document which one was used. |
| `remediation` | ✓ (array of steps) | ✓ | **Keep**, both agree — array of Thai-language step strings. |
| `confidence` | ✗ (diagram instead has `ensemble_confidence` on the upstream `ml_out` node, and separate `faithfulness_score`/`hallucination_rate` on `llm_out`) | ✓ | **Add, but as a pass-through, not an LLM self-report.** `src/modeling/fusion_layer.py`'s `fuse()` returns only `fused_score`/`fused_prediction` — no confidence scalar. The calling code (this phase's module, not the LLM) must derive `confidence` deterministically from the Fusion Layer's own internals before it ever reaches the prompt — e.g. `rf_confidence = abs(rf_norm - 0.5) * 2` (already computed inside `fuse_confidence_weighted`, just not currently returned) or `abs(fused_score - FUSED_DECISION_THRESHOLD)` normalized to `[0,1]`. **Which exact formula is not decided here — see Open Questions.** What *is* decided: `confidence` must be a numeric value computed by ML-pipeline code and inserted into the prompt as a fact the LLM must copy/reference verbatim (same discipline as IPs/ports — see the few-shot below), never a number the LLM is asked to invent about its own explanation. |
| `faithfulness_score` / `hallucination_rate` | ✓, listed as if part of every `llm_out` | — | **Excluded from the live per-alert JSON.** A real scoring method against evaluation ground truth has not been defined, so these values must not be fabricated or derived from contract pass rates. The user deferred metric production until that method is defined. |
| `response_ms` | ✓ | — | **Keep**, but computed by the calling code via wall-clock timing around the LLM HTTP call, never asked of the model. |

**Final live per-alert JSON schema** (what `src/llm/llm_explain.py`'s entry point returns):
```json
{
  "priority": "HIGH",
  "explanation_th": "...",
  "technique_id": "T1046",
  "mitre_tactic": "discovery",
  "affected_asset": "10.20.14.87",
  "remediation": ["...", "...", "..."],
  "confidence": 0.91,
  "response_ms": 2400
}
```

### Prompt design and few-shot entity preservation — concrete example, implement to this spec
The prompt must instruct the model, in Thai, to copy technical identifiers (IPs, ports, hostnames, `technique_id` values) **verbatim** into `explanation_th` and every other field — never translate, transliterate, round, or paraphrase them. Below is one concrete few-shot pair the prompt template must include (build at least 2 such pairs spanning different categories — see Deliverables):

**Input (alert context assembled from `ml_out`-equivalent + RAG retrieval, all real fields this phase's code actually has available):**
```json
{
  "source_ip": "10.20.14.87",
  "dst_ip": "192.168.40.12",
  "affected_asset": "FS-PRINT02.campus.local",
  "dst_port": 445,
  "timestamp": "2026-07-14T09:12:33+07:00",
  "rf_predicted_label": "PortScan",
  "fused_score": 0.91,
  "fused_prediction": 1,
  "confidence": 0.91,
  "rag_top_chunk": {
    "technique_id": "T1046",
    "chunk_type": "detection",
    "tactic": "discovery",
    "document": "Behavioral Detection Strategy for Network Service Discovery Across Platforms - Analytic 1057: Detects processes performing network enumeration (e.g., port scans, service probing) by correlating process creation, socket connections, and sequential destination IP probing within a time window."
  }
}
```

**Expected output (Thai JSON, per the final schema above):**
```json
{
  "priority": "HIGH",
  "explanation_th": "ตรวจพบพฤติกรรมการสแกนพอร์ต (Network Service Discovery) จากต้นทาง 10.20.14.87 ไปยัง FS-PRINT02.campus.local ที่พอร์ต 445 เมื่อเวลา 2026-07-14T09:12:33+07:00 โมเดล Random Forest จำแนกเป็น PortScan และ Fusion Layer ให้คะแนนรวม fused_score=0.91 ซึ่งตรงกับเทคนิค MITRE ATT&CK T1046 ในกลวิธี discovery — ลักษณะการเชื่อมต่อจำนวนมากไปยังปลายทางเดียวกันในช่วงเวลาสั้น ๆ สอดคล้องกับการสแกนบริการเครือข่ายเพื่อหาช่องโหว่",
  "technique_id": "T1046",
  "mitre_tactic": "discovery",
  "affected_asset": "FS-PRINT02.campus.local",
  "remediation": [
    "บล็อกการเชื่อมต่อจากต้นทาง 10.20.14.87 ที่ไฟร์วอลล์ทันที",
    "ตรวจสอบ log การเชื่อมต่อขาเข้าที่พอร์ต 445 บน FS-PRINT02.campus.local ย้อนหลัง 24 ชั่วโมง",
    "ประสานทีมเครือข่ายเพื่อตรวจสอบว่ามีการเคลื่อนไหวด้านข้าง (lateral movement) ตามมาหลังจากการสแกนนี้หรือไม่"
  ],
  "confidence": 0.91,
  "response_ms": 2400
}
```
Note in this example: `10.20.14.87`, `445`, `FS-PRINT02.campus.local`, `T1046`, `0.91` all appear byte-for-byte identical between input and output — none are translated, reformatted, or paraphrased. This exact preservation property is what Deliverable §3's entity-preservation tests assert.

### Reuse this repo's "decision gets a config sidecar" discipline
Following `models/fusion_config.json` and `models/rag_retrieval_config.json`'s precedent: every decision made in this phase (backend choice, model tag, quantization level, GPU offload configuration, the schema reconciliation table above, the `confidence` formula once picked) must be written to a JSON sidecar (`models/llm_layer_config.json`), not left only in code comments or this handoff.

### No leakage, extended to this phase
`config/mitre_mapping.csv`'s `technique_name`/`tactic` columns must never be fed into a live production prompt (mirrors `src/rag/rag_retrieval.py`'s and `src/evaluation/eval_rag_precision.py`'s existing no-leakage rule — only `RagRetriever.retrieve()`'s actual output and the Fusion Layer's actual output are legitimate inference-time signals). `config/mitre_mapping.csv` is eval-ground-truth only, read solely by `src/evaluation/eval_llm_layer.py`.

## Deliverables

1. **`src/llm/llm_explain.py`** — the inference-time module, following `src/modeling/fusion_layer.py`/`src/rag/rag_retrieval.py`'s standalone-importable-module pattern. At minimum: `load_env_config()` (LLM_* keys, fail-fast on missing, mirroring `rag_retrieval.load_env_config`), `build_prompt(alert_context, rag_hits, fused_score, fused_prediction, confidence) -> str`, `call_llm(prompt) -> str` (HTTP call to the local Ollama/llama.cpp server), `parse_llm_response(raw_text) -> dict` (must handle malformed/non-JSON model output defensively — do not assume the model always returns valid JSON; consider requesting Ollama's `format: "json"` constrained-decoding mode or an equivalent llama.cpp grammar, and document whichever choice was made and why in `models/llm_layer_config.json`), and one entry point, e.g. `explain_alert(alert_context, rag_hits, fused_score, fused_prediction) -> dict` returning exactly the schema above.
2. **A prompt template** — either embedded as a module-level constant string in `src/llm/llm_explain.py` (there is no existing `prompts/` directory precedent in this repo to follow, so don't invent one unless you have a good reason) containing the Thai system/instruction text plus the ≥2 few-shot pairs (one is fully specified above; author at least one more spanning a different category, e.g. an SSH-Patator/`T1110.001` example).
3. **`models/llm_layer_config.json`** — chosen backend (`ollama`|`llamacpp`), verified model tag/quantization, GPU offload configuration and measured tokens/sec, the full schema-reconciliation table above (or a reference to this handoff plus any deviations), and the `confidence` formula actually implemented.
4. **`src/evaluation/eval_llm_layer.py`** — mirroring `src/evaluation/eval_rag_precision.py`'s shape (read a fixed test-case set, call the module under test, write per-case + aggregate results to a CSV, no single blended number). Must implement, at minimum, these **named test cases**:

   | Test ID | Category | Alert fields | RAG input (real, from `reports/rag_retrieval_precision_at_3.csv`) | Assertion |
   |---|---|---|---|---|
   | `TC-ENT-01` | PortScan (entity preservation) | `source_ip=10.20.14.87`, `dst_port=445`, `affected_asset=FS-PRINT02.campus.local` | RAG top-3 for "PortScan" (real hit, precision@3=0.67): `T1046::detection`, `mitigation::M1016`, `mitigation::M1031` | `explanation_th` contains the exact substrings `"10.20.14.87"`, `"445"`, `"FS-PRINT02.campus.local"` unmodified; `technique_id == "T1046"` present verbatim somewhere in the JSON. |
   | `TC-ENT-02` | SSH-Patator (entity preservation, second example, different field values) | `source_ip=203.0.113.44`, `dst_port=22`, `affected_asset=AUTH-SVR01.campus.local` | RAG top-3 for "SSH-Patator" (real hit, precision@3=0.33): `T1110.001::detection`, `T1059::detection`, `mitigation::M1037` | Same verbatim-substring checks as `TC-ENT-01`, on different values — guards against a prompt that happens to work for one example but generalizes to translating/rounding a different IP/port. |
   | `TC-FAITH-01` | DoS Hulk (grounded/normal) | representative flood-alert fields | RAG top-3 for "DoS Hulk" (real hit, precision@3=0.67, `n_relevant_in_top3=2`): `T1498::description`, `T1499.002::detection`, `T1499.002::description` | `technique_id` output is one of `{"T1498", "T1499.002"}` (i.e. matches a chunk actually retrieved, not an unrelated technique never surfaced); `mitre_tactic` matches that technique's real tactic (`T1499.002`'s is `impact`, confirm in `data/knowledge_base/mitre_techniques.json` before hardcoding); `explanation_th` does not assert a technique/tactic contradicting the retrieved chunk text. |
   | `TC-FAITH-02` | PortScan (grounded/normal, second example) | as `TC-ENT-01`'s alert fields | same RAG input as `TC-ENT-01` | `technique_id == "T1046"` (the only relevant chunk actually retrieved); `mitre_tactic == "discovery"`. |
   | `TC-ADV-01` | **Bot (adversarial/known-gap)** | representative Bot-category alert fields, `rf_predicted_label="Bot"` | RAG top-3 for "Bot" (real **miss**, precision@3=0.0, `n_relevant_in_top3=0`, per `reports/rag_retrieval_precision_at_3.csv`): `T1046::detection` (technique_id `T1046`), `T1499.002::detection` (`T1499.002`), `T1498::detection` (`T1498`) — **none is the true `T1071.001`** | See Open Questions — the correct fallback behavior is not yet decided, so this test's assertion is a placeholder to be filled in once decided (candidate behaviors: refuse to state a `technique_id` with false confidence / explicitly hedge in `explanation_th` / defer to `rf_predicted_label` alone without RAG-sourced justification). What **is** already decided: the assertion must fail the test if the LLM confidently states `technique_id` as one of `T1046`/`T1499.002`/`T1498` (the wrong retrieved ones) framed as certain/well-grounded, since none of those chunks are actually relevant to Bot/C2-beaconing behavior. |
   | `TC-ADV-02` | **Heartbleed (adversarial/known-gap)** | representative Heartbleed-category alert fields, `rf_predicted_label` per whatever the ensemble actually outputs for this category | RAG top-3 for "Heartbleed" (real **miss**, precision@3=0.0, `n_relevant_in_top3=0`): `T1071.001::description` (`T1071.001`), `T1499.002::detection` (`T1499.002`), `T1110::description` (`T1110`) — **none is the true `T1190`** | Same shape as `TC-ADV-01`: must not confidently state `technique_id` as one of the wrongly-retrieved ones; exact required fallback behavior deferred to Open Questions. |

   Output: `reports/llm_layer_eval_results.csv` (per-test-case pass/fail + any measured faithfulness/entity-preservation sub-scores, aggregate row at the bottom — mirroring `reports/rag_retrieval_precision_at_3.csv`'s per-category-then-aggregate shape) plus a plain-text `reports/llm_layer_report.txt` summarizing the run, following `reports/rag_layer_report.txt`'s section-numbered convention.

   **Deferred by user (2026-10-16)**: do not write a `soc_llm_eval_summary` point until a real faithfulness and hallucination scoring method against evaluation ground truth has been defined. Contract pass rates are not substitutes. The dashboard marks the metric as paused in the meantime.

## Definition of done

- [ ] Typhoon2.1 12B Instruct actually runs locally (via the chosen backend) and returns a real completion for at least one full-length representative prompt (not a toy prompt) — measured, not assumed, tokens/sec and VRAM usage recorded in `models/llm_layer_config.json`.
- [ ] If full GPU offload didn't fit in 8GB VRAM, the actual configuration used (partial offload layer count, or CPU fallback) is documented, not silently glossed over.
- [ ] `src/llm/llm_explain.py` runs end-to-end from a fresh process/kernel restart with no dependency on notebook-only in-memory state (same bar `src/rag/rag_retrieval.py` was held to).
- [ ] `explain_alert(...)`'s return value matches the reconciled schema exactly (all 8 fields present, `mitre_tactic` is a real lowercase tactic string not a `Txxxx` value, `confidence` is pipeline-derived not model-self-reported, `faithfulness_score`/`hallucination_rate` are **absent** from this per-alert output).
- [ ] `src/llm/llm_explain.py` never feeds `config/mitre_mapping.csv`'s `technique_name`/`tactic` columns into a live prompt (no-leakage check, mirroring the RAG phase's own check).
- [ ] All 6 named test cases (`TC-ENT-01/02`, `TC-FAITH-01/02`, `TC-ADV-01/02`) are implemented in `src/evaluation/eval_llm_layer.py` and produce a real pass/fail result (not stubbed with `pass`/`TODO` left in place, except where Open Questions below explicitly says the assertion itself is pending a decision).
- [ ] `reports/llm_layer_eval_results.csv` and `reports/llm_layer_report.txt` exist with real (not placeholder) output from an actual run.
- [x] Faithfulness metric production is deferred until its ground-truth scoring method is defined; no synthetic or contract-pass-rate substitute is written.
- [ ] `models/llm_layer_config.json` documents every decision this handoff left to the developer (backend, model tag/quantization, GPU config, `confidence` formula, JSON-validity enforcement mechanism), the same way `models/rag_retrieval_config.json` documents the RAG phase's own decisions.
- [ ] `code-reviewer` has been run against the finished work before flagging it done — this is not optional, per this repo's established norm (see `reports/rag_layer_report.txt` §7 for what that review actually caught last time: two documentation-accuracy corrections, not code bugs — don't assume a clean pass means no findings are worth recording).

## Open questions / things NOT to decide silently

1. **The exact `confidence` formula.** This handoff decided *that* `confidence` must be pipeline-derived (not LLM-self-reported) but not *which* formula — `rf_confidence = abs(rf_norm - 0.5) * 2` (reusing `fuse_confidence_weighted`'s internal notion of "confidence," which is really RF's decisiveness, not correctness — note `models/fusion_config.json`'s own `known_limitations` warn that RF can be confident and wrong zero-day) vs. `abs(fused_score - FUSED_DECISION_THRESHOLD)` normalized, vs. something else. Flag this and ask before hardcoding one.
2. **The correct fallback behavior for `TC-ADV-01`/`TC-ADV-02` (Bot/Heartbleed).** This handoff explicitly does not decide what the LLM *should* do when RAG retrieval is confirmed wrong (defer to `rf_predicted_label` with a hedge? Refuse to name a `technique_id` at all and only report `priority`/`remediation` from the ML side? Something else?) — this is a real product/thesis-methodology decision (how should the system fail safely when its own retrieval layer is known-wrong for a category), not an implementation detail, and per this project's own established norm of surfacing ambiguity rather than silently resolving it, it must be asked about before the two adversarial test cases' assertions can be finalized.
3. **Whether `technique_id` should ever come from `config/mitre_mapping.csv`'s per-row `mapping_confidence`/RF's predicted label at all, vs. RAG top-1 only.** The reconciliation table above says `technique_id` is sourced from "the RAG retriever's top-1 hit... or the Fusion Layer's/RF's own predicted category... as a secondary signal" but does not specify a tie-break/precedence rule when they disagree (which, per the Bot/Heartbleed gap, they will, routinely, for at least 2 of 14 categories). Decide and document this precedence rule before implementation, don't let it fall out of whatever the code happens to do first.
4. **Exact Typhoon2.1 12B Instruct source/tag.** Not verified in this brief (this session's tools were research/planning tools only, not used to actually reach out to Ollama's library or HuggingFace) — if `ollama pull` (or an equivalent GGUF search) doesn't turn up a ready artifact under an obvious tag, stop and report rather than silently substituting a different Typhoon variant/size or a different model entirely.
5. **JSON-validity enforcement mechanism** (constrained decoding via Ollama's `format: "json"` / a llama.cpp GBNF grammar / prompt-only instruction + a parse-and-retry loop) — pick one, document the choice and its failure-mode handling in `models/llm_layer_config.json`, but if none of these actually work reliably with the chosen backend/model, report that rather than shipping a module that silently crashes or returns `None` on the first malformed completion.
