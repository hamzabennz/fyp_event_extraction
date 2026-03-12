"""
main_batches.py
---------------
Runs event extraction over batches of email .txt files from txt_emails/,
appending all extracted events to EVENTS.json incrementally.

Usage:
    python main_batches.py [--batch-size N] [--start-from M] [--max-emails K]

Options:
    --batch-size N   Number of emails per batch sent to the LLM (default: 10)
    --start-from M   Skip the first M emails (resume from a checkpoint, default: 0)
    --max-emails K   Stop after processing K emails in total (default: all)
"""

import json
import os
import argparse
import time
from typing import List, Dict
from dotenv import load_dotenv
import google.generativeai as genai
from smolagents import CodeAgent, Tool, ChatMessage, Model

# Load environment variables from .env file
load_dotenv()

# ==========================================
# CONFIG
# ==========================================
EMAIL_DIR       = "txt_emails"
EVENTS_FILE     = "EVENTS.json"
NARRATIVE_FILE  = "EVENTS_NARRATIVE.txt"
PROGRESS_FILE   = "batch_progress.json"   # tracks how far we've gotten
EVENT_DB_FILE   = "event_types_db.json"

DEFAULT_BATCH_SIZE = 10    # emails per LLM call
RETRY_LIMIT        = 3     # retries per batch on failure
RETRY_DELAY        = 5     # seconds between retries


# ==========================================
# 1. LOAD THE EVENT TYPES SCHEMA
# ==========================================
try:
    with open(EVENT_DB_FILE, "r") as f:
        event_schema = json.load(f)
    print(f"✅ Event Type Database loaded from {EVENT_DB_FILE}")
except FileNotFoundError:
    print(f"❌ Error: {EVENT_DB_FILE} not found.")
    exit(1)

event_types_list = ", ".join(event_schema.keys())
event_schema_formatted = json.dumps(event_schema, separators=(',', ':'))

event_extraction_guide = "\n\n".join([
    f"EVENT TYPE: {event_type}\n"
    f"Description: {event_schema[event_type]['description']}\n"
    f"Required Fields to Extract:\n" +
    "\n".join([
        f"  - {field_name}: {field_desc}"
        for field_name, field_desc in event_schema[event_type]['specific_fields'].items()
    ])
    for event_type in event_schema.keys()
])

SYSTEM_PROMPT = f"""
You are an Event Extraction AI Assistant specialized in Digital Forensics. Your task is to:
1. Analyze raw text evidence logs to identify and extract events based on the provided schema.
2. For each event found, extract: date_time, location, parties involved, and type-specific fields.
3. Crucially, provide a JUSTIFICATION for why this event was extracted, citing specific phrases from the text.
4. Generate a COMPREHENSIVE NARRATIVE that describes the event thoroughly, including all non-N/A fields.
5. Return extracted events as a properly formatted JSON list ONLY. No extra text.

AVAILABLE EVENT TYPES AND EXTRACTION GUIDELINES:
{event_extraction_guide}

EVENT SCHEMA (Full JSON):
{event_schema_formatted}

IMPORTANT: For each extracted event, ensure you create a JSON object with this EXACT structure:
{{
  "type": "one_of_the_event_types_above",
  "justification": "Explain WHY you extracted this. Quote the exact text snippet that proves this event occurred.",
  "confidence_score": "High/Medium/Low",
  "date_time": "extracted_date_and_time",
  "location": "extracted_location",
  "parties": ["person1", "person2"],
  "narrative": "A comprehensive forensic narrative. Structure: [Who] [Action] [When] [Where] [How/Platform] [Specific Details]. DO NOT use N/A values. Omit fields that are N/A. Be thorough.",
  "type_specific_fields": {{
    "field_name_1": "extracted_value",
    "field_name_2": "extracted_value"
  }}
}}

RULES:
- Do NOT extract hypothetical or proposed events.
- If a field is missing, use "N/A" in type_specific_fields.
- Match the type_specific_fields keys exactly as defined in the schema.
- If NO events are found in the text, return an empty JSON array: []
- Return ONLY a valid JSON array. No markdown, no explanation, no code fences.
"""


# ==========================================
# 2. GEMINI MODEL WRAPPER
# ==========================================
class GeminiModel(Model):
    def __init__(self, model_name="gemini-2.0-flash", api_key=None, **kwargs):
        super().__init__(model_id=model_name, **kwargs)
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def __call__(self, messages, stop_sequences=None, **kwargs) -> ChatMessage:
        return self.generate(messages, stop_sequences=stop_sequences, **kwargs)

    def generate(self, messages, stop_sequences=None, **kwargs) -> ChatMessage:
        prompt_parts = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content", "")
            else:
                role = getattr(m, "role", "user")
                content = getattr(m, "content", str(m))

            if role == "system":
                prompt_parts.append(f"System Instruction: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Model: {content}")
            else:
                prompt_parts.append(f"{role}: {content}")

        full_prompt = "\n".join(prompt_parts)
        try:
            response = self.model.generate_content(full_prompt)
            return ChatMessage(role="assistant", content=response.text)
        except Exception as e:
            return ChatMessage(role="assistant", content=f"Error: {str(e)}")


# ==========================================
# 3. HELPERS
# ==========================================

def load_existing_events() -> List[Dict]:
    """Load events already saved to EVENTS.json, or return empty list."""
    if os.path.exists(EVENTS_FILE):
        try:
            with open(EVENTS_FILE, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, Exception):
            pass
    return []


def save_events(events: List[Dict]):
    """Overwrite EVENTS.json with the full list and regenerate EVENTS_NARRATIVE.txt."""
    with open(EVENTS_FILE, "w") as f:
        json.dump(events, f, indent=2)

    narratives = []
    for i, event in enumerate(events, 1):
        narrative_text = event.get("narrative", "No narrative available")
        confidence = event.get("confidence_score", "N/A")
        narratives.append(f"--- Event {i} ---\n{narrative_text}\n[Confidence: {confidence}]\n")

    with open(NARRATIVE_FILE, "w") as f:
        f.write("\n".join(narratives))


def load_progress() -> int:
    """Return the index of the last successfully processed email (0-based)."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                return json.load(f).get("last_processed_index", -1)
        except Exception:
            pass
    return -1


def save_progress(last_index: int, total_events: int):
    """Checkpoint: save how far we've gotten."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "last_processed_index": last_index,
            "total_events_so_far": total_events
        }, f, indent=2)


def get_sorted_email_files() -> List[str]:
    """Return all email_N.txt files sorted numerically."""
    files = [f for f in os.listdir(EMAIL_DIR) if f.startswith("email_") and f.endswith(".txt")]
    files.sort(key=lambda x: int(x.replace("email_", "").replace(".txt", "")))
    return files


def extract_json_from_response(text: str) -> List[Dict]:
    """
    Robustly extract a JSON array from LLM response text.
    Handles markdown code fences and extra surrounding text.
    """
    if text is None:
        return []

    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])          # drop first line (``` or ```json)
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]        # drop trailing ```

    # Find the outermost JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []

    json_str = text[start:end+1]
    try:
        result = json.loads(json_str)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    return []


def call_gemini_direct(model_obj: GeminiModel, batch_text: str) -> List[Dict]:
    """
    Call Gemini directly (bypassing smolagents) to extract events from a batch.
    Returns a list of event dicts.
    """
    prompt = f"""{SYSTEM_PROMPT}

---
Extract all events from the following email batch.
Return ONLY a valid JSON array of events. If no events found, return [].

EMAIL BATCH:
{batch_text}
"""
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = model_obj.model.generate_content(prompt)
            raw_text = response.text if response and hasattr(response, "text") else ""
            events = extract_json_from_response(raw_text)
            return events
        except Exception as e:
            print(f"      ⚠️  Attempt {attempt}/{RETRY_LIMIT} failed: {e}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY * attempt)
    print(f"      ❌ All {RETRY_LIMIT} attempts failed for this batch. Skipping.")
    return []


# ==========================================
# 4. MAIN BATCH LOOP
# ==========================================

def run_batches(batch_size: int, start_from: int, max_emails: int, resume: bool):
    print("\n" + "="*60)
    print("🚀 Batch Event Extraction — main_batches.py")
    print("="*60)

    # Build file list
    all_files = get_sorted_email_files()
    total_available = len(all_files)
    print(f"📁 Found {total_available} email files in '{EMAIL_DIR}/'")

    # Apply --start-from
    all_files = all_files[start_from:]

    # Apply --max-emails
    if max_emails > 0:
        all_files = all_files[:max_emails]

    print(f"📋 Processing {len(all_files)} emails "
          f"(starting at index {start_from}, batch size {batch_size})")

    # Resume from checkpoint if requested
    checkpoint_index = -1
    if resume:
        checkpoint_index = load_progress()
        if checkpoint_index >= 0:
            # checkpoint is the absolute index; convert to relative
            relative_skip = checkpoint_index - start_from + 1
            if relative_skip > 0 and relative_skip < len(all_files):
                print(f"⏩ Resuming from checkpoint: skipping first {relative_skip} files "
                      f"(last processed: email index {checkpoint_index})")
                all_files = all_files[relative_skip:]
            else:
                print(f"⏩ Checkpoint at index {checkpoint_index} — already done or out of range.")

    # Load existing events
    all_events = load_existing_events()
    print(f"📦 Existing events loaded: {len(all_events)}\n")

    # Initialize model
    model_obj = GeminiModel()

    # Split into batches
    batches = [all_files[i:i+batch_size] for i in range(0, len(all_files), batch_size)]
    total_batches = len(batches)

    for batch_num, batch_files in enumerate(batches, 1):
        print(f"\n{'─'*50}")
        print(f"📨 Batch {batch_num}/{total_batches} — {len(batch_files)} emails")

        # Read and concatenate email texts
        batch_text_parts = []
        for fname in batch_files:
            fpath = os.path.join(EMAIL_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read().strip()
                batch_text_parts.append(f"=== {fname} ===\n{content}")
            except Exception as e:
                print(f"   ⚠️  Could not read {fname}: {e}")

        if not batch_text_parts:
            print("   ⚠️  No readable emails in batch. Skipping.")
            continue

        batch_text = "\n\n".join(batch_text_parts)

        # Extract events
        print(f"   🔍 Extracting events...")
        new_events = call_gemini_direct(model_obj, batch_text)
        print(f"   ✅ Extracted {len(new_events)} events from this batch.")

        # Append and persist
        all_events.extend(new_events)
        save_events(all_events)

        # Update checkpoint: last processed is the last file in this batch (absolute index)
        last_file_name = batch_files[-1]  # e.g. "email_45.txt"
        last_abs_index = int(last_file_name.replace("email_", "").replace(".txt", "")) - 1
        save_progress(last_index=start_from + (batch_num * batch_size) - 1,
                      total_events=len(all_events))

        print(f"   💾 Total events so far: {len(all_events)} — checkpoint saved.")

        # Small delay to avoid rate limits
        if batch_num < total_batches:
            time.sleep(1)

    print("\n" + "="*60)
    print(f"🏁 Done! Total events extracted: {len(all_events)}")
    print(f"   📄 EVENTS.json       — {len(all_events)} events")
    print(f"   📄 EVENTS_NARRATIVE.txt — updated")
    print(f"   📄 {PROGRESS_FILE}   — final checkpoint")
    print("="*60)


# ==========================================
# 5. ENTRY POINT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch event extraction from txt_emails/ using Gemini."
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Number of emails per batch (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--start-from", type=int, default=0,
        help="Skip the first N emails in sorted order (default: 0)"
    )
    parser.add_argument(
        "--max-emails", type=int, default=0,
        help="Maximum number of emails to process in total (0 = all, default: 0)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the last saved checkpoint in batch_progress.json"
    )
    args = parser.parse_args()

    run_batches(
        batch_size=args.batch_size,
        start_from=args.start_from,
        max_emails=args.max_emails,
        resume=args.resume,
    )
