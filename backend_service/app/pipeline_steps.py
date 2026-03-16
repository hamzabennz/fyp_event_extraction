from __future__ import annotations

import asyncio
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import google.generativeai as genai


LogFn = Callable[[str], None]


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
    batch_size: int = 10,
    retry_limit: int = 3,
) -> list[dict]:
    event_schema = _load_event_schema(root_dir)
    system_prompt = _build_extraction_prompt(event_schema)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required to run extraction")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    batches = [evidence_files[i : i + batch_size] for i in range(0, len(evidence_files), batch_size)]
    all_events: list[dict] = []

    for batch_index, batch in enumerate(batches, start=1):
        log(f"Extraction batch {batch_index}/{len(batches)} with {len(batch)} evidence file(s)")
        text_parts: list[str] = []
        evidence_names: list[str] = []
        for path in batch:
            content = path.read_text(encoding="utf-8", errors="replace").strip()
            text_parts.append(f"=== SOURCE: evidence/{path.name} ===\n{content}")
            evidence_names.append(f"evidence/{path.name}")
        batch_text = "\n\n".join(text_parts)

        prompt = f"""{system_prompt}

---
Extract all events from the following evidence batch.
Each evidence item is preceded by a header like "=== SOURCE: evidence/file.txt ===".
Use that path as source_file for events extracted from that file.

EVIDENCE BATCH:
{batch_text}
"""

        extracted: list[dict] = []
        for attempt in range(1, retry_limit + 1):
            try:
                response = model.generate_content(prompt)
                raw_text = response.text if response and hasattr(response, "text") else ""
                extracted = _extract_json_from_response(raw_text)
                break
            except Exception as error:
                log(f"Extraction retry {attempt}/{retry_limit} failed: {error}")
                if attempt < retry_limit:
                    time.sleep(2 * attempt)

        for event in extracted:
            if "source_file" not in event or not event.get("source_file"):
                if len(evidence_names) == 1:
                    event["source_file"] = evidence_names[0]
                else:
                    snippet = event.get("snippet", "") or event.get("justification", "")
                    matched = "UNKNOWN"
                    if snippet:
                        head = snippet[:80]
                        for name, part in zip(evidence_names, text_parts):
                            if head in part:
                                matched = name
                                break
                    event["source_file"] = matched
            all_events.append(event)

    for idx, event in enumerate(all_events):
        event["id"] = idx

    write_event_artifacts(output_dir, all_events)
    log(f"Extraction complete with {len(all_events)} event(s)")
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
    from google import genai as google_genai
    from text_lloom.llm import EmbedModel, Model

    def setup_llm_fn(api_key_value: str):
        return google_genai.Client(api_key=api_key_value)

    def setup_embed_fn(api_key_value: str):
        return google_genai.Client(api_key=api_key_value)

    async def call_llm_fn(model, prompt):
        if "system_prompt" not in model.args:
            model.args["system_prompt"] = "You are a helpful assistant who helps with identifying patterns in text examples."
        if "temperature" not in model.args:
            model.args["temperature"] = 0
        config = {
            "temperature": model.args["temperature"],
            "max_output_tokens": 65536,
        }
        if "JSON" in prompt or "json" in prompt:
            config["response_mime_type"] = "application/json"

        for attempt in range(5):
            try:
                result = model.client.models.generate_content(model=model.name, contents=prompt, config=config)
                return (result.text if result and hasattr(result, "text") else None), [0, 0]
            except Exception:
                time.sleep(2 * (attempt + 1))
        return None, [0, 0]

    def call_embed_fn(model, text_arr):
        if isinstance(text_arr, str):
            text_arr = [text_arr]
        valid_indices = [i for i, value in enumerate(text_arr) if value and isinstance(value, str) and value.strip()]
        if not valid_indices:
            return [[0.0] * 3072] * len(text_arr), [0, 0]

        filtered_text = [text_arr[i] for i in valid_indices]
        embeddings_map = {}
        for i in range(0, len(filtered_text), 10):
            batch = filtered_text[i : i + 10]
            for attempt in range(3):
                try:
                    result = model.client.models.embed_content(model="gemini-embedding-001", contents=batch)
                    if hasattr(result, "embeddings") and result.embeddings:
                        for j, emb in enumerate(result.embeddings):
                            embeddings_map[valid_indices[i + j]] = emb.values
                    break
                except Exception:
                    time.sleep(2 * (attempt + 1))

        vectors = [embeddings_map.get(i, [0.0] * 3072) for i in range(len(text_arr))]
        return vectors, [0, 0]

    models = {
        "distill_model": Model(
            setup_fn=setup_llm_fn,
            fn=call_llm_fn,
            name="gemini-2.0-flash",
            cost=[0.0, 0.0],
            rate_limit=(1000, 1000),
            context_window=32000,
            api_key=api_key,
        ),
        "cluster_model": EmbedModel(
            setup_fn=setup_embed_fn,
            fn=call_embed_fn,
            name="gemini-embedding-001",
            cost=(0.00001 / 1000),
            batch_size=10,
            api_key=api_key,
        ),
        "synth_model": Model(
            setup_fn=setup_llm_fn,
            fn=call_llm_fn,
            name="gemini-2.0-flash",
            cost=[0.01 / 1000, 0.03 / 1000],
            rate_limit=(60, 60),
            context_window=32000,
            api_key=api_key,
        ),
        "score_model": Model(
            setup_fn=setup_llm_fn,
            fn=call_llm_fn,
            name="gemini-2.5-flash",
            cost=[0.0005 / 1000, 0.0015 / 1000],
            rate_limit=(60, 60),
            context_window=32000,
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

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required to run LLooM")

    async def _async_run() -> pd.DataFrame:
        import sys

        lloom_src = root_dir / "lloom" / "text_lloom" / "src"
        if str(lloom_src) not in sys.path:
            sys.path.insert(0, str(lloom_src))

        import text_lloom.workbench as wb

        df = pd.read_csv(output_dir / "events.csv")
        if len(df) < 15:
            multiplier = (20 // len(df)) + 1 if len(df) > 0 else 1
            df = pd.concat([df] * multiplier, ignore_index=True)

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
        score_df = await l.score(debug=False, batch_size=50, get_highlights=False)
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
                    & (score_df_combined["score"] >= 0.75)
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
                covered_by_generic_ids = pivot_generic[pivot_generic >= 0.75].index.tolist()
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
                score_df2 = await l2.score(debug=False, batch_size=50, get_highlights=False)
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
                log(line.strip())
    if stderr:
        for line in stderr.splitlines():
            if line.strip():
                log(f"stderr: {line.strip()}")

    if process.returncode != 0:
        raise RuntimeError(f"Script failed ({script_path.name}) with exit code {process.returncode}")
