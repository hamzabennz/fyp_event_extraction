from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Callable

_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FFFF"
    "\U00002600-\U000027BF"
    "\U0000FE00-\U0000FE0F"
    "\U00020000-\U0002A6DF"
    "]",
    flags=re.UNICODE,
)


def _strip_emojis(text: str) -> str:
    return _EMOJI_RE.sub("", text).strip()

import pandas as pd
import numpy as np
import google.generativeai as genai
from openai import OpenAI as _OpenAI
from sentence_transformers import SentenceTransformer

LogFn = Callable[[str], None]


def _gemini_call_with_retry(model, prompt: str, retry_limit: int, log: LogFn) -> str:
    """Call a Gemini model with retry/backoff, returning raw text or empty string."""
    from config import CONFIG
    backoff_base = CONFIG.extraction.gemini_retry_backoff_base_seconds
    for attempt in range(1, retry_limit + 1):
        try:
            response = model.generate_content(prompt)
            return response.text if response and hasattr(response, "text") else ""
        except Exception as error:
            log(f"Gemini call attempt {attempt}/{retry_limit} failed: {error}")
            if attempt < retry_limit:
                time.sleep(backoff_base * attempt)
    return ""


def _deepseek_call_with_retry(client: _OpenAI, model_name: str, prompt: str, retry_limit: int, log: LogFn) -> str:
    """Call DeepSeek via OpenAI-compatible SDK with retry/backoff."""
    from config import CONFIG
    backoff_base = CONFIG.extraction.gemini_retry_backoff_base_seconds
    for attempt in range(1, retry_limit + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            return response.choices[0].message.content if response.choices else ""
        except Exception as error:
            log(f"DeepSeek call attempt {attempt}/{retry_limit} failed: {error}")
            if attempt < retry_limit:
                time.sleep(backoff_base * attempt)
    return ""


def _omniroute_call_with_retry(client: _OpenAI, model_name: str, prompt: str, retry_limit: int, log: LogFn) -> str:
    """Call OmniRoute via OpenAI-compatible SDK with retry/backoff."""
    from config import CONFIG
    backoff_base = CONFIG.extraction.gemini_retry_backoff_base_seconds
    for attempt in range(1, retry_limit + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            return response.choices[0].message.content if response.choices else ""
        except Exception as error:
            log(f"OmniRoute call attempt {attempt}/{retry_limit} failed: {error}")
            if attempt < retry_limit:
                time.sleep(backoff_base * attempt)
    return ""


def _load_event_schema(root_dir: Path) -> dict:
    schema_path = root_dir / "event_types_db.json"
    with schema_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_event_artifacts(output_dir: Path, events: list[dict], log: LogFn | None = None) -> None:
    events_json_path = output_dir / "EVENTS.json"
    events_json_path.write_text(json.dumps(events, indent=2), encoding="utf-8")

    narrative_lines = []
    for i, event in enumerate(events, start=1):
        narrative = event.get("narrative", "No narrative available")
        confidence = event.get("confidence_score", "N/A")
        narrative_lines.append(f"--- Event {i} ---\n{narrative}\n[Confidence: {confidence}]\n")
    (output_dir / "EVENTS_NARRATIVE.txt").write_text("\n".join(narrative_lines), encoding="utf-8")

    if log is not None:
        log(f"Persisted {len(events)} reviewed event(s) to artifacts")


def _extract_json_from_response(text: str | None) -> list[dict]:
    if text is None:
        return []

    clean = text.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:])
        if clean.rstrip().endswith("```"):
            clean = clean.rstrip()[:-3]

    start = clean.find("[")
    end = clean.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []

    payload = clean[start : end + 1]
    try:
        parsed = json.loads(payload)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        return []
    return []


# ─── Hybrid pipeline helpers ─────────────────────────────────────────────────

def _cosine_similarity(v1, v2) -> float:
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / denom) if denom > 0 else 0.0


_st_model: SentenceTransformer | None = None


def _get_st_model() -> SentenceTransformer:
    """Lazily load the sentence-transformer model (shared across calls)."""
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


def _get_top_k_event_types(
    trigger_text: str,
    sentence_context: str,
    type_names: list[str],
    type_embeddings: np.ndarray,
    k: int = 5,
) -> list[str]:
    """Return the top-k event-type names most similar to a trigger in context."""
    st = _get_st_model()
    query = f"Trigger '{trigger_text}' in context: {sentence_context}"
    query_emb = st.encode(query)
    sims = [
        (_cosine_similarity(query_emb, type_embeddings[i]), name)
        for i, name in enumerate(type_names)
    ]
    sims.sort(key=lambda x: x[0], reverse=True)
    return [name for _, name in sims[:k]]


def _build_focused_extraction_prompt(event_schema: dict, top_event_types: list[str]) -> str:
    """Build a system prompt narrowed to only the given event types."""
    filtered = {k: event_schema[k] for k in top_event_types if k in event_schema}
    schema_json = json.dumps(filtered, separators=(",", ":"))
    guide = "\n\n".join(
        f"EVENT TYPE: {et}\n"
        f"Description: {filtered[et]['description']}\n"
        "Required Fields to Extract:\n"
        + "\n".join(
            f"  - {fn}: {fd}"
            for fn, fd in filtered[et]["specific_fields"].items()
        )
        for et in filtered
    )
    return f"""You are a highly meticulous Digital Forensics Event Extraction AI. Your goal is EXHAUSTIVE RECALL.
Your task is to analyze raw text evidence logs to identify and extract EVERY SINGLE EVENT based on the provided schema.

INSTRUCTIONS FOR HIGH RECALL:
1. Read the text sentence by sentence.
2. Analyze the timeline of events chronologically (who did what, when, and where).
3. If an event occurs multiple times, extract EACH ONE as a separate event.
4. Do NOT extract hypothetical or proposed events.
5. Provide a JUSTIFICATION citing specific phrases from the text.
6. Generate a COMPREHENSIVE NARRATIVE including all non-N/A fields.

AVAILABLE EVENT TYPES AND EXTRACTION GUIDELINES:
{guide}

EVENT SCHEMA (Full JSON):
{schema_json}

OUTPUT FORMAT INSTRUCTIONS:
Step 1: Write an "Analysis Scratchpad" in plain text. List the chronological timeline briefly.
***CRITICAL RULE FOR SCRATCHPAD: Do NOT use square brackets. Use parentheses instead.***

Step 2: After your scratchpad, return extracted events as a properly formatted JSON array ONLY.

For each event use this EXACT structure:
[{{
    "type": "one_of_the_event_types_above",
    "justification": "Explain WHY you extracted this event.",
    "snippet": "EXACT verbatim text from the source document.",
    "confidence_score": "High/Medium/Low",
    "date_time": "extracted_date_and_time",
    "location": "extracted_location",
    "parties": ["person1", "person2"],
    "narrative": "Comprehensive forensic narrative.",
    "source_file": "evidence/<filename>.txt",
    "type_specific_fields": {{
        "field_name_1": "extracted_value"
    }}
}}]

RULES:
- Do NOT extract hypothetical or proposed events.
- If a field is missing, use \"N/A\" in type_specific_fields.
- Match type_specific_fields keys exactly as defined in the schema.
- If NO events are found, return []
""".strip()


def _detect_triggers_remote(remote_url: str, sentences: list[str], log: LogFn) -> list[dict]:
    """Run GLEN trigger detection via a remote HTTP endpoint (e.g. a Kaggle-hosted
    GPU instance exposed through ngrok). See GLEN_REMOTE_URL in env.example.
    """
    import requests

    url = remote_url.rstrip("/") + "/detect"
    log(f"Running GLEN trigger detection via remote endpoint ({url})")
    try:
        response = requests.post(url, json={"sentences": sentences}, timeout=300)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        raise RuntimeError(f"GLEN_REMOTE_URL request to {url} failed: {error}") from error


def _detect_triggers_hybrid(
    root_dir: Path,
    sentences: list[str],
    log: LogFn,
) -> list[dict]:
    """Run GLEN trigger detection on a list of sentences.

    Returns a list of {sentence, triggers} dicts, same format as
    custom_trigger_detection.detect_triggers.
    Falls back to returning all sentences with a dummy trigger if GLEN is
    unavailable (so extraction still runs, just without trigger filtering).
    """
    remote_url = os.getenv("GLEN_REMOTE_URL")
    if remote_url:
        return _detect_triggers_remote(remote_url, sentences, log)

    ckpt_path = root_dir / "GLEN" / "ckpts"
    if not ckpt_path.exists():
        raise RuntimeError(f"GLEN checkpoint not found at {ckpt_path}. Cannot run hybrid extraction.")

    import sys
    glen_dir = str(root_dir / "GLEN")
    original_cwd = os.getcwd()
    try:
        os.chdir(glen_dir)
        if glen_dir not in sys.path:
            sys.path.insert(0, glen_dir)
        from custom_trigger_detection import detect_triggers  # type: ignore
        return detect_triggers(sentences, "ckpts")
    finally:
        os.chdir(original_cwd)


def _build_extraction_prompt(event_schema: dict) -> str:
    event_schema_formatted = json.dumps(event_schema, separators=(",", ":"))
    event_extraction_guide = "\n\n".join(
        [
            f"EVENT TYPE: {event_type}\n"
            f"Description: {event_schema[event_type]['description']}\n"
            f"Required Fields to Extract:\n"
            + "\n".join(
                [
                    f"  - {field_name}: {field_desc}"
                    for field_name, field_desc in event_schema[event_type]["specific_fields"].items()
                ]
            )
            for event_type in event_schema.keys()
        ]
    )

    return f"""
You are an Event Extraction AI Assistant specialized in Digital Forensics. Your task is to:
1. Analyze raw evidence text logs to identify and extract events based on the provided schema.
2. For each event found, extract: date_time, location, parties involved, and type-specific fields.
3. Provide a JUSTIFICATION for why this event was extracted.
4. Generate a COMPREHENSIVE NARRATIVE that describes the event thoroughly, including all non-N/A fields.
5. Return extracted events as a properly formatted JSON list ONLY. No extra text.

AVAILABLE EVENT TYPES AND EXTRACTION GUIDELINES:
{event_extraction_guide}

EVENT SCHEMA (Full JSON):
{event_schema_formatted}

IMPORTANT: Return each event using this structure:
{{
  "type": "one_of_the_event_types_above",
  "justification": "Explain WHY this is an event",
  "snippet": "EXACT verbatim text proving this event occurred",
  "confidence_score": "High/Medium/Low",
  "date_time": "extracted_date_and_time",
  "location": "extracted_location",
  "parties": ["party_1", "party_2"],
  "narrative": "Comprehensive forensic narrative with all non-N/A fields",
  "type_specific_fields": {{
    "field_name_1": "value",
    "field_name_2": "value"
  }},
  "source_file": "evidence/<filename>.txt"
}}

RULES:
- Do NOT extract hypothetical or proposed events.
- If a field is missing, use "N/A" in type_specific_fields.
- Match type_specific_fields keys exactly as defined in schema.
- If NO events are found in text, return []
- Return ONLY a valid JSON array.
""".strip()


def extract_events_from_evidence(
    *,
    root_dir: Path,
    evidence_files: list[Path],
    output_dir: Path,
    log: LogFn,
    batch_size: int | None = None,
    retry_limit: int | None = None,
) -> list[dict]:
    from config import CONFIG, get_llm_provider
    retry_limit = retry_limit if retry_limit is not None else CONFIG.extraction.gemini_call_retry_limit

    event_schema = _load_event_schema(root_dir)
    provider = get_llm_provider()

    # ── Init LLM client ────────────────────────────────────────────────────────
    if provider == "deepseek":
        ds_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not ds_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek")
        ds_client = _OpenAI(api_key=ds_api_key, base_url="https://api.deepseek.com")
        ds_model_name = CONFIG.models.deepseek_extraction_model
        log(f"Using DeepSeek for extraction (model: {ds_model_name})")
    elif provider == "omniroute":
        or_api_key = os.getenv("OMNIROUTE_API_KEY")
        if not or_api_key:
            raise RuntimeError("OMNIROUTE_API_KEY is required when LLM_PROVIDER=omniroute")
        or_base_url = os.getenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1")
        or_client = _OpenAI(api_key=or_api_key, base_url=or_base_url)
        or_model_name = CONFIG.models.omniroute_extraction_model
        log(f"Using OmniRoute for extraction (model: {or_model_name}, base_url: {or_base_url})")
    else:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required to run extraction")
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(CONFIG.models.extraction_model)
        log(f"Using Gemini for extraction (model: {CONFIG.models.extraction_model})")

    def _call_llm(prompt: str) -> str:
        if provider == "deepseek":
            return _deepseek_call_with_retry(ds_client, ds_model_name, prompt, retry_limit, log)
        if provider == "omniroute":
            return _omniroute_call_with_retry(or_client, or_model_name, prompt, retry_limit, log)
        return _gemini_call_with_retry(gemini_model, prompt, retry_limit, log)

    # ── Step 1: Read all evidence files into sentences ─────────────────────────
    log("Hybrid extraction: splitting evidence into sentences")
    sentences: list[str] = []
    sentence_source_map: dict[str, str] = {}  # sentence → source filename

    for path in evidence_files:
        content = path.read_text(encoding="utf-8", errors="replace").strip()
        for raw_s in content.split("."):
            cleaned = raw_s.strip().replace("\n", " ")
            if cleaned:
                if not cleaned.endswith((".", "?", "!")):
                    cleaned += "."
                # Use index-based key to avoid collisions when two files share
                # identical sentence text.
                sentence_source_map[len(sentences)] = f"evidence/{path.name}"
                sentences.append(cleaned)

    if not sentences:
        log("No sentences extracted — skipping extraction")
        write_event_artifacts(output_dir, [], log)
        return []

    log(f"Split {len(evidence_files)} file(s) into {len(sentences)} sentence(s)")

    # ── Step 2: GLEN trigger detection ────────────────────────────────────────
    log("Running GLEN trigger detection...")
    triggers_per_sentence = _detect_triggers_hybrid(root_dir, sentences, log)

    # ── Step 3: Embed event types once for similarity ranking ─────────────────
    log("Embedding event type descriptions for similarity ranking...")
    st = _get_st_model()
    event_type_names = list(event_schema.keys())
    event_type_descriptions = [
        f"{name}: {event_schema[name]['description']}" for name in event_type_names
    ]
    type_embeddings = st.encode(event_type_descriptions)

    # ── Step 4: Per-sentence focused extraction ───────────────────────────────
    all_events: list[dict] = []
    processed = 0

    for sent_idx, item in enumerate(triggers_per_sentence):
        sent = item["sentence"]
        triggers = item["triggers"]

        high_conf = [t for t in triggers if t.get("confidence", 0) > 0.8]
        if not high_conf:
            continue

        processed += 1
        source_file = sentence_source_map.get(sent_idx, "UNKNOWN")

        # Collect top-K candidate event types across all triggers in this sentence
        candidate_types: set[str] = set()
        for t in high_conf:
            top_k = _get_top_k_event_types(t["text"], sent, event_type_names, type_embeddings, k=5)
            candidate_types.update(top_k)

        candidate_list = list(candidate_types)
        log(f"Sentence {processed}: {len(high_conf)} trigger(s) → types: {candidate_list}")

        focused_prompt = _build_focused_extraction_prompt(event_schema, candidate_list)
        prompt = f"""{focused_prompt}

---
Extract all events from the following evidence sentence.
Source file for all events in this sentence: {source_file}
Return ONLY a valid JSON array. If no events found, return [].

EVIDENCE:
{sent}
"""
        raw_text = _call_llm(prompt)
        extracted = _extract_json_from_response(raw_text)

        for event in extracted:
            if not event.get("source_file"):
                event["source_file"] = source_file
        all_events.extend(extracted)

    log(f"Hybrid extraction complete: {processed} sentence(s) processed, {len(all_events)} event(s) found")

    for idx, event in enumerate(all_events):
        event["id"] = idx

    write_event_artifacts(output_dir, all_events, log)
    return all_events


def build_csv_from_events(output_dir: Path, log: LogFn) -> None:
    events_path = output_dir / "EVENTS.json"
    if not events_path.exists():
        raise RuntimeError("EVENTS.json not found before CSV generation")

    events = json.loads(events_path.read_text(encoding="utf-8"))
    events_csv = output_dir / "events.csv"
    enriched_csv = output_dir / "events_enriched.csv"

    with events_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "event"])
        writer.writeheader()
        for i, event in enumerate(events):
            writer.writerow({"id": event.get("id", i), "event": event.get("narrative", "No narrative available")})

    with enriched_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "event", "source_file", "snippet"])
        writer.writeheader()
        for i, event in enumerate(events):
            writer.writerow(
                {
                    "id": event.get("id", i),
                    "event": event.get("narrative", "No narrative available"),
                    "source_file": event.get("source_file", "UNKNOWN"),
                    "snippet": event.get("snippet", ""),
                }
            )

    log(f"Generated CSV artifacts from {len(events)} event(s)")


def _prepare_lloom_models(api_key: str):
    from config import CONFIG
    from google import genai as google_genai
    from text_lloom.llm import EmbedModel, Model

    lc = CONFIG.lloom
    m  = CONFIG.models

    def setup_llm_fn(api_key_value: str):
        return google_genai.Client(api_key=api_key_value)

    def setup_embed_fn(api_key_value: str):
        return google_genai.Client(api_key=api_key_value)

    async def call_llm_fn(model, prompt):
        if "system_prompt" not in model.args:
            model.args["system_prompt"] = "You are a helpful assistant who helps with identifying patterns in text examples."
        if "temperature" not in model.args:
            model.args["temperature"] = lc.llm_temperature
        gen_config = {
            "temperature": model.args["temperature"],
            "max_output_tokens": lc.llm_max_output_tokens,
        }
        if "JSON" in prompt or "json" in prompt:
            gen_config["response_mime_type"] = "application/json"

        for attempt in range(lc.llm_call_retry_limit):
            try:
                result = model.client.models.generate_content(model=model.name, contents=prompt, config=gen_config)
                return (result.text if result and hasattr(result, "text") else None), [0, 0]
            except Exception:
                time.sleep(2 * (attempt + 1))
        return None, [0, 0]

    def call_embed_fn(model, text_arr):
        if isinstance(text_arr, str):
            text_arr = [text_arr]
        zero_vec = [0.0] * lc.embedding_output_dimension
        valid_indices = [i for i, value in enumerate(text_arr) if value and isinstance(value, str) and value.strip()]
        if not valid_indices:
            return [zero_vec] * len(text_arr), [0, 0]

        filtered_text = [text_arr[i] for i in valid_indices]
        embeddings_map = {}
        for i in range(0, len(filtered_text), lc.embedding_batch_size):
            batch = filtered_text[i : i + lc.embedding_batch_size]
            for attempt in range(lc.embed_call_retry_limit):
                try:
                    result = model.client.models.embed_content(model=m.lloom_embedding_model, contents=batch)
                    if hasattr(result, "embeddings") and result.embeddings:
                        for j, emb in enumerate(result.embeddings):
                            embeddings_map[valid_indices[i + j]] = emb.values
                    break
                except Exception:
                    time.sleep(2 * (attempt + 1))

        vectors = [embeddings_map.get(i, zero_vec) for i in range(len(text_arr))]
        return vectors, [0, 0]

    models = {
        "distill_model": Model(
            setup_fn=setup_llm_fn,
            fn=call_llm_fn,
            name=m.lloom_distill_model,
            cost=[0.0, 0.0],
            rate_limit=lc.distill_model_rate_limit_rpm_tpm,
            context_window=lc.llm_context_window_tokens,
            api_key=api_key,
        ),
        "cluster_model": EmbedModel(
            setup_fn=setup_embed_fn,
            fn=call_embed_fn,
            name=m.lloom_embedding_model,
            cost=(0.00001 / 1000),
            batch_size=lc.embedding_batch_size,
            api_key=api_key,
        ),
        "synth_model": Model(
            setup_fn=setup_llm_fn,
            fn=call_llm_fn,
            name=m.lloom_concept_synthesis_model,
            cost=[0.01 / 1000, 0.03 / 1000],
            rate_limit=lc.synth_and_score_model_rate_limit_rpm_tpm,
            context_window=lc.llm_context_window_tokens,
            api_key=api_key,
        ),
        "score_model": Model(
            setup_fn=setup_llm_fn,
            fn=call_llm_fn,
            name=m.lloom_scoring_model,
            cost=[0.0005 / 1000, 0.0015 / 1000],
            rate_limit=lc.synth_and_score_model_rate_limit_rpm_tpm,
            context_window=lc.llm_context_window_tokens,
            api_key=api_key,
        ),
    }
    return models


def _prepare_lloom_models_deepseek(api_key: str):
    """Build LLooM model wrappers that call DeepSeek (LLM) + sentence-transformers (embeddings)."""
    from config import CONFIG
    from openai import AsyncOpenAI
    from sentence_transformers import SentenceTransformer
    from text_lloom.llm import EmbedModel, Model

    lc = CONFIG.lloom
    m  = CONFIG.models

    deepseek_async_client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    embed_dim = lc.deepseek_embedding_output_dimension

    def setup_fn(api_key_value):
        return None  # clients captured in closures

    async def call_llm_fn(model, prompt):
        system_msg = model.args.get("system_prompt", "You are a helpful assistant who helps with identifying patterns in text examples.")
        for attempt in range(lc.llm_call_retry_limit):
            try:
                response = await deepseek_async_client.chat.completions.create(
                    model=m.deepseek_lloom_model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=lc.llm_max_output_tokens,
                    temperature=lc.llm_temperature,
                )
                text = response.choices[0].message.content if response.choices else None
                return text, [0, 0]
            except Exception:
                await asyncio.sleep(2 * (attempt + 1))
        return None, [0, 0]

    def call_embed_fn(model, text_arr):
        if isinstance(text_arr, str):
            text_arr = [text_arr]
        zero_vec = [0.0] * embed_dim
        results = []
        for t in text_arr:
            if t and isinstance(t, str) and t.strip():
                results.append(st_model.encode(t).tolist())
            else:
                results.append(zero_vec)
        return results, [0, 0]

    models = {
        "distill_model": Model(
            setup_fn=setup_fn,
            fn=call_llm_fn,
            name=m.deepseek_lloom_model,
            cost=[0.0, 0.0],
            rate_limit=lc.distill_model_rate_limit_rpm_tpm,
            context_window=lc.llm_context_window_tokens,
            api_key=api_key,
        ),
        "cluster_model": EmbedModel(
            setup_fn=setup_fn,
            fn=call_embed_fn,
            name="all-MiniLM-L6-v2",
            cost=0.0,
            batch_size=lc.embedding_batch_size,
            api_key=api_key,
        ),
        "synth_model": Model(
            setup_fn=setup_fn,
            fn=call_llm_fn,
            name=m.deepseek_lloom_model,
            cost=[0.0, 0.0],
            rate_limit=lc.synth_and_score_model_rate_limit_rpm_tpm,
            context_window=lc.llm_context_window_tokens,
            api_key=api_key,
        ),
        "score_model": Model(
            setup_fn=setup_fn,
            fn=call_llm_fn,
            name=m.deepseek_lloom_model,
            cost=[0.0, 0.0],
            rate_limit=lc.synth_and_score_model_rate_limit_rpm_tpm,
            context_window=lc.llm_context_window_tokens,
            api_key=api_key,
        ),
    }
    return models


def _prepare_lloom_models_omniroute(api_key: str):
    """Build LLooM model wrappers that call OmniRoute (LLM) + sentence-transformers (embeddings)."""
    from config import CONFIG
    from openai import AsyncOpenAI
    from sentence_transformers import SentenceTransformer
    from text_lloom.llm import EmbedModel, Model

    lc = CONFIG.lloom
    m  = CONFIG.models

    base_url = os.getenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1")
    omniroute_async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    embed_dim = lc.deepseek_embedding_output_dimension

    def setup_fn(api_key_value):
        return None  # clients captured in closures

    async def call_llm_fn(model, prompt):
        system_msg = model.args.get("system_prompt", "You are a helpful assistant who helps with identifying patterns in text examples.")
        for attempt in range(lc.llm_call_retry_limit):
            try:
                response = await omniroute_async_client.chat.completions.create(
                    model=m.omniroute_lloom_model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=lc.llm_max_output_tokens,
                    temperature=lc.llm_temperature,
                )
                text = response.choices[0].message.content if response.choices else None
                return text, [0, 0]
            except Exception:
                await asyncio.sleep(2 * (attempt + 1))
        return None, [0, 0]

    def call_embed_fn(model, text_arr):
        if isinstance(text_arr, str):
            text_arr = [text_arr]
        zero_vec = [0.0] * embed_dim
        results = []
        for t in text_arr:
            if t and isinstance(t, str) and t.strip():
                results.append(st_model.encode(t).tolist())
            else:
                results.append(zero_vec)
        return results, [0, 0]

    models = {
        "distill_model": Model(
            setup_fn=setup_fn,
            fn=call_llm_fn,
            name=m.omniroute_lloom_model,
            cost=[0.0, 0.0],
            rate_limit=lc.distill_model_rate_limit_rpm_tpm,
            context_window=lc.llm_context_window_tokens,
            api_key=api_key,
        ),
        "cluster_model": EmbedModel(
            setup_fn=setup_fn,
            fn=call_embed_fn,
            name="all-MiniLM-L6-v2",
            cost=0.0,
            batch_size=lc.embedding_batch_size,
            api_key=api_key,
        ),
        "synth_model": Model(
            setup_fn=setup_fn,
            fn=call_llm_fn,
            name=m.omniroute_lloom_model,
            cost=[0.0, 0.0],
            rate_limit=lc.synth_and_score_model_rate_limit_rpm_tpm,
            context_window=lc.llm_context_window_tokens,
            api_key=api_key,
        ),
        "score_model": Model(
            setup_fn=setup_fn,
            fn=call_llm_fn,
            name=m.omniroute_lloom_model,
            cost=[0.0, 0.0],
            rate_limit=lc.synth_and_score_model_rate_limit_rpm_tpm,
            context_window=lc.llm_context_window_tokens,
            api_key=api_key,
        ),
    }
    return models


def _run_lloom_mock(output_dir: Path, log: LogFn) -> None:
    events_csv = output_dir / "events.csv"
    if not events_csv.exists():
        raise RuntimeError("events.csv not found before mock LLooM scoring")

    df = pd.read_csv(events_csv)
    columns = [
        "doc_id",
        "text",
        "concept_id",
        "concept_name",
        "concept_prompt",
        "score",
        "rationale",
        "highlight",
        "concept_seed",
    ]

    if df.empty:
        pd.DataFrame(columns=columns).to_csv(output_dir / "score_results_combined.csv", index=False)
        log("Mock LLooM mode enabled: events.csv is empty, wrote empty scoring output")
        return

    concepts = [
        (
            "mock-timeline-pattern",
            "Timeline Escalation",
            "Does this evidence indicate meaningful escalation in timeline-critical activity?",
        ),
        (
            "mock-coordination-pattern",
            "Coordination Signals",
            "Does this evidence show coordination between parties around a shared objective?",
        ),
        (
            "mock-operational-pattern",
            "Operational Planning",
            "Does this evidence include operational planning details, logistics, or execution signals?",
        ),
    ]

    records: list[dict] = []
    for _, row in df.iterrows():
        doc_id = str(int(row["id"])) if pd.notna(row.get("id")) else "0"
        text = str(row.get("event", "") or "")
        if not text.strip():
            text = "No narrative available"

        try:
            dominant_index = int(doc_id) % len(concepts)
        except ValueError:
            dominant_index = 0

        for concept_index, (concept_id, concept_name, concept_prompt) in enumerate(concepts):
            if concept_index == dominant_index:
                score = 0.92
                rationale = f"Mock strong match for {concept_name.lower()}"
            elif concept_index == (dominant_index + 1) % len(concepts):
                score = 0.58
                rationale = f"Mock partial relevance to {concept_name.lower()}"
            else:
                score = 0.14
                rationale = f"Mock weak relevance to {concept_name.lower()}"

            records.append(
                {
                    "doc_id": doc_id,
                    "text": text,
                    "concept_id": concept_id,
                    "concept_name": concept_name,
                    "concept_prompt": concept_prompt,
                    "score": score,
                    "rationale": rationale,
                    "highlight": "",
                    "concept_seed": "",
                }
            )

    pd.DataFrame(records, columns=columns).to_csv(output_dir / "score_results_combined.csv", index=False)
    log(
        "Mock LLooM mode enabled: generated "
        f"{len(records)} scoring rows across {len(concepts)} concepts for {len(df)} event(s)"
    )


def run_lloom_iterative(
    *,
    root_dir: Path,
    output_dir: Path,
    log: LogFn,
    max_concepts: int = 5,
    max_iterations: int = 3,
    generic_coverage_threshold: float = 0.5,
    mock_mode: bool = False,
) -> None:
    if mock_mode:
        _run_lloom_mock(output_dir, log)
        return

    from config import CONFIG, get_llm_provider
    provider = get_llm_provider()

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek")
        log(f"Using DeepSeek + sentence-transformers for LLooM (model: {CONFIG.models.deepseek_lloom_model})")
    elif provider == "omniroute":
        api_key = os.getenv("OMNIROUTE_API_KEY")
        if not api_key:
            raise RuntimeError("OMNIROUTE_API_KEY is required when LLM_PROVIDER=omniroute")
        log(f"Using OmniRoute + sentence-transformers for LLooM (model: {CONFIG.models.omniroute_lloom_model})")
    else:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required to run LLooM")
        log(f"Using Gemini for LLooM (model: {CONFIG.models.lloom_distill_model})")

    async def _async_run() -> pd.DataFrame:
        import sys

        lloom_src = root_dir / "lloom" / "text_lloom" / "src"
        if str(lloom_src) not in sys.path:
            sys.path.insert(0, str(lloom_src))

        import text_lloom.workbench as wb

        from config import CONFIG as _cfg
        lc = _cfg.lloom
        df = pd.read_csv(output_dir / "events.csv")
        if len(df) < lc.min_rows_required_for_induction:
            multiplier = (lc.min_rows_required_for_induction // len(df)) + 1 if len(df) > 0 else 1
            df = pd.concat([df] * multiplier, ignore_index=True)

        if provider == "deepseek":
            models = _prepare_lloom_models_deepseek(api_key)
        elif provider == "omniroute":
            models = _prepare_lloom_models_omniroute(api_key)
        else:
            models = _prepare_lloom_models(api_key)

        l = wb.lloom(
            df=df,
            text_col="event",
            id_col="id",
            distill_model=models["distill_model"],
            cluster_model=models["cluster_model"],
            synth_model=models["synth_model"],
            score_model=models["score_model"],
        )

        custom_prompts = {
            "distill_filter": None,
            "distill_summarize": None,
            "synthesize": None,
        }

        log("LLooM iteration 1: generating concepts")
        await l.gen(custom_prompts=custom_prompts, auto_review=True, debug=False)
        await l.select_auto(max_concepts=max_concepts)
        score_df = await l.score(debug=False, batch_size=lc.scoring_batch_size, get_highlights=False)
        score_df_combined = score_df.copy()

        remaining_outliers = 0
        reached_max_with_outliers = False

        for iteration in range(2, max_iterations + 1):
            concept_names_all = [c for c in score_df_combined["concept_name"].unique() if c != "Outlier"]
            if not concept_names_all:
                break

            total_events = score_df_combined["doc_id"].nunique()
            generic_concepts: list[str] = []
            for concept_name in concept_names_all:
                matched = score_df_combined[
                    (score_df_combined["concept_name"] == concept_name)
                    & (score_df_combined["score"] >= lc.generic_concept_min_coverage_score)
                ]["doc_id"].nunique()
                if total_events > 0 and (matched / total_events) >= generic_coverage_threshold:
                    generic_concepts.append(concept_name)

            non_generic = [c for c in concept_names_all if c not in generic_concepts]
            if non_generic:
                pivot = score_df_combined[score_df_combined["concept_name"].isin(non_generic)].groupby("doc_id")["score"].max()
            else:
                pivot = score_df_combined[score_df_combined["concept_name"].isin(concept_names_all)].groupby("doc_id")["score"].max()

            outlier_doc_ids = pivot[pivot == 0.0].index.tolist()

            if generic_concepts:
                pivot_generic = score_df_combined[score_df_combined["concept_name"].isin(generic_concepts)].groupby("doc_id")["score"].max()
                covered_by_generic_ids = pivot_generic[pivot_generic >= lc.generic_concept_min_coverage_score].index.tolist()
                covered_by_generic_ids = [doc_id for doc_id in covered_by_generic_ids if doc_id in outlier_doc_ids]
            else:
                covered_by_generic_ids = []

            all_loop_ids = list(set(outlier_doc_ids + covered_by_generic_ids))
            remaining_outliers = len(all_loop_ids)
            if not all_loop_ids:
                log(f"LLooM stopped after {iteration-1} iteration(s): no uncovered outliers")
                break

            in_df_copy = l.in_df.copy()
            in_df_copy[l.doc_id_col] = in_df_copy[l.doc_id_col].astype(str)
            outlier_df = in_df_copy[in_df_copy[l.doc_id_col].isin(all_loop_ids)].reset_index(drop=True)
            if outlier_df.empty:
                break

            log(f"LLooM iteration {iteration}: rerunning on {len(outlier_df)} uncovered evidence rows")
            l2 = wb.lloom(
                df=outlier_df,
                text_col=l.doc_col,
                id_col=l.doc_id_col,
                distill_model=l.distill_model,
                cluster_model=l.cluster_model,
                synth_model=l.synth_model,
                score_model=l.score_model,
            )
            try:
                await l2.gen(custom_prompts=custom_prompts, auto_review=True, debug=False)
                await l2.select_auto(max_concepts=max_concepts)

                l2.in_df = l.in_df
                l2.df_to_score = l.in_df
                score_df2 = await l2.score(debug=False, batch_size=lc.scoring_batch_size, get_highlights=False)
            except Exception as error:
                error_text = str(error)
                if "k >= N" in error_text or "scipy.linalg.eigh" in error_text:
                    log(
                        "LLooM outlier rerun stopped early due to small-subset spectral decomposition limit "
                        f"({error_text}). Keeping concepts from completed iterations."
                    )
                    break
                raise

            for concept_id, concept in l2.concepts.items():
                if concept.active:
                    l.concepts[concept_id] = concept

            score_df_combined = pd.concat([score_df_combined, score_df2], ignore_index=True)
            if iteration == max_iterations and remaining_outliers > 0:
                reached_max_with_outliers = True

        if reached_max_with_outliers:
            log(
                f"LLooM reached max_iterations={max_iterations}; "
                f"{remaining_outliers} outlier evidence row(s) may remain uncovered"
            )

        return score_df_combined

    combined = asyncio.run(_async_run())
    combined.to_csv(output_dir / "score_results_combined.csv", index=False)
    log(f"LLooM scoring complete with {len(combined)} rows")


def run_python_script(
    script_path: Path,
    working_dir: Path,
    log: LogFn,
    timeout_seconds: int = 900,
    cancel_check: Callable[[], bool] | None = None,
) -> None:
    process = subprocess.Popen(
        ["python", str(script_path)],
        cwd=str(working_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    start = time.time()

    while process.poll() is None:
        if cancel_check and cancel_check():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            raise RuntimeError(f"Execution cancelled while running {script_path.name}")

        if timeout_seconds > 0 and (time.time() - start) > timeout_seconds:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            raise RuntimeError(f"Timeout reached ({timeout_seconds}s) while running {script_path.name}")

        time.sleep(1)

    stdout, stderr = process.communicate()
    if stdout:
        for line in stdout.splitlines():
            if line.strip():
                log(_strip_emojis(line.strip()))
    if stderr:
        for line in stderr.splitlines():
            if line.strip():
                log(f"stderr: {_strip_emojis(line.strip())}")

    if process.returncode != 0:
        raise RuntimeError(f"Script failed ({script_path.name}) with exit code {process.returncode}")
