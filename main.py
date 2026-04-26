import json
import os
import time
from typing import List, Dict

from dotenv import load_dotenv
import google.generativeai as genai
from smolagents import ChatMessage, Model

# Load environment variables from .env file
load_dotenv()

# ==========================================
# 1. LOAD THE EVENT TYPES SCHEMA
# ==========================================
EVENT_DB_FILE = "event_types_db.json"

# Load event type definitions from file
try:
    with open(EVENT_DB_FILE, "r") as f:
        event_schema = json.load(f)
    print(f"✅ Event Type Database loaded from {EVENT_DB_FILE}")
except FileNotFoundError:
    print(f"❌ Error: {EVENT_DB_FILE} not found. Please ensure the file exists.")
    exit(1)


EVENTS_FILE = "EVENTS.json"
NARRATIVE_FILE = "EVENTS_NARRATIVE.txt"
RETRY_LIMIT = 3
RETRY_DELAY = 5


# ==========================================
# 2. DEFINE THE MODEL WRAPPER
# ==========================================
class GeminiModel(Model):
    """Wrapper for Google Gemini used with direct generation calls."""

    def __init__(self, model_name="gemini-3-flash-preview", api_key=None, **kwargs):
        super().__init__(model_id=model_name, **kwargs)
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in .env file.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def __call__(self, messages: List[Dict[str, str]], stop_sequences: List[str] = None, **kwargs) -> ChatMessage:
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
            return ChatMessage(role="assistant", content=f"Error generating response: {str(e)}")

# ==========================================
# 4. INITIALIZE AGENT
# ==========================================

# System prompt for the agent - dynamically built from event schema
event_types_list = ", ".join(event_schema.keys())
# Minify JSON schema to single line to save tokens
event_schema_formatted = json.dumps(event_schema, separators=(',', ':'))

# Build detailed event extraction instructions
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
You are a highly meticulous Digital Forensics Event Extraction AI. Your goal is EXHAUSTIVE RECALL.
Your task is to analyze raw text evidence logs to identify and extract EVERY SINGLE EVENT based on the provided schema.

INSTRUCTIONS FOR HIGH RECALL:
1. Read the text sentence by sentence.
2. Analyze the timeline of events chronologically (who did what, when, and where).
3. If an event occurs multiple times (e.g., multiple payments), extract EACH ONE as a separate event.
4. Do NOT extract hypothetical or proposed events unless they are confirmed to have happened.
5. Crucially, provide a JUSTIFICATION for why this event was extracted, citing specific phrases from the text.
6. Generate a COMPREHENSIVE NARRATIVE that describes the event thoroughly, including all non-N/A fields.

AVAILABLE EVENT TYPES AND EXTRACTION GUIDELINES:
{event_extraction_guide}

EVENT SCHEMA (Full JSON):
{event_schema_formatted}

OUTPUT FORMAT INSTRUCTIONS:
Step 1: Write an "Analysis Scratchpad" in plain text at the very top of your response. Briefly list out the chronological timeline of events you see in the text. Thinking step-by-step guarantees no event is missed.
***CRITICAL RULE FOR SCRATCHPAD: Do NOT use square brackets "[" or "]" anywhere in your scratchpad. Use parentheses "(" and ")" instead.***

Step 2: After your scratchpad, return the extracted events as a properly formatted JSON array ONLY.

IMPORTANT: For each extracted event, ensure you create a JSON object inside the array with this EXACT structure:[
    {{
        "type": "one_of_the_event_types_above",
        "justification": "Explain WHY you extracted this event. Describe the reasoning, not the raw quote.",
        "snippet": "Copy-paste the EXACT verbatim text span from the source document that proves this event occurred. Do NOT paraphrase.",
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
]

NOTE ON "snippet" vs "justification":
- "snippet" = the EXACT verbatim text copied from the source. This will be used for text highlighting.
- "justification" = your reasoning for why this is an event. Keep them separate.

RULES:
- Do NOT extract hypothetical or proposed events.
- If a field is missing, use "N/A" in type_specific_fields.
- Match the type_specific_fields keys exactly as defined in the schema.
- If NO events are found in the text, return an empty JSON array: []

RULES FOR EXTRACTION:
- Do NOT extract hypothetical or proposed events (e.g., "We should go to Austin") unless they are confirmed to have happened.
- If a field is missing, use "N/A" in type_specific_fields.
- Match the type_specific_fields keys exactly as defined in the schema.
- For the NARRATIVE field: Write comprehensively, not poetically. Include all non-N/A fields. Write like a forensic report, not a story.
"""

def load_existing_events() -> List[Dict]:
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
    for i, event in enumerate(events):
        event["id"] = i

    with open(EVENTS_FILE, "w") as f:
        json.dump(events, f, indent=2)

    narratives = []
    for i, event in enumerate(events, 1):
        narrative_text = event.get("narrative", "No narrative available")
        confidence = event.get("confidence_score", "N/A")
        narratives.append(f"--- Event {i} ---\n{narrative_text}\n[Confidence: {confidence}]\n")

    with open(NARRATIVE_FILE, "w") as f:
        f.write("\n".join(narratives))


def extract_json_from_response(text: str) -> List[Dict]:
    if text is None:
        return []

    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []

    json_str = text[start:end + 1]
    try:
        result = json.loads(json_str)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    return []


def call_gemini_direct(model_obj: GeminiModel, evidence_text: str) -> List[Dict]:
    prompt = f"""{SYSTEM_PROMPT}

---
Extract all events from the following evidence text.
Return ONLY a valid JSON array of events. If no events found, return [].

EVIDENCE:
{evidence_text}
"""

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = model_obj.model.generate_content(prompt)
            raw_text = response.text if response and hasattr(response, "text") else ""
            return extract_json_from_response(raw_text)
        except Exception as e:
            print(f"      ⚠️  Attempt {attempt}/{RETRY_LIMIT} failed: {e}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY * attempt)

    print(f"      ❌ All {RETRY_LIMIT} attempts failed. Skipping.")
    return []


def main():
    raw_evidence = ""
    try:
        with open("emails2.txt", "r") as f:
            raw_evidence = f.read()
        print("✅ Successfully loaded raw evidence from emails2.txt")
    except FileNotFoundError:
        print("❌ Error: emails2.txt not found. Please ensure the file exists.")
        exit(1)

    print("--- STARTING EVENT EXTRACTION ---")

    model = GeminiModel()
    existing_events = load_existing_events()
    if existing_events:
        print(f"📦 Existing events loaded: {len(existing_events)}")

    new_events = call_gemini_direct(model, raw_evidence)
    print(f"✅ Extracted {len(new_events)} events")

    all_events = existing_events + new_events
    save_events(all_events)

    print("\n--- EXTRACTION COMPLETE ---")
    print(f"📄 EVENTS.json       — {len(all_events)} events")
    print(f"📄 EVENTS_NARRATIVE.txt — updated")


if __name__ == "__main__":
    main()