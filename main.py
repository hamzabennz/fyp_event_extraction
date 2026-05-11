import json
import os
import time
from typing import List, Dict

from dotenv import load_dotenv
import google.genai as genai
from smolagents import ChatMessage, Model
import sys

import huggingface_hub
if not hasattr(huggingface_hub, 'is_offline_mode'):
    huggingface_hub.is_offline_mode = lambda: False

# Setup for sentence-transformers to calculate similarities
from sentence_transformers import SentenceTransformer
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Add GLEN directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "GLEN"))

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

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

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
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
            )
            return ChatMessage(role="assistant", content=response.text)
        except Exception as e:
            return ChatMessage(role="assistant", content=f"Error generating response: {str(e)}")


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def build_system_prompt(top_event_types: List[str]) -> str:
    # Filter the schema for only the top types
    filtered_schema = {k: event_schema[k] for k in top_event_types if k in event_schema}
    event_schema_formatted = json.dumps(filtered_schema, separators=(',', ':'))

    event_extraction_guide = "\n\n".join([
        f"EVENT TYPE: {event_type}\n"
        f"Description: {filtered_schema[event_type]['description']}\n"
        f"Required Fields to Extract:\n" +
        "\n".join([
            f"  - {field_name}: {field_desc}"
            for field_name, field_desc in filtered_schema[event_type]['specific_fields'].items()
        ])
        for event_type in filtered_schema.keys()
    ])

    return f"""You are a highly meticulous Digital Forensics Event Extraction AI. Your goal is EXHAUSTIVE RECALL.
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


def call_gemini_direct(model_obj: GeminiModel, evidence_text: str, system_prompt: str) -> List[Dict]:
    prompt = f"""{system_prompt}

---
Extract all events from the following evidence text.
Return ONLY a valid JSON array of events. If no events found, return [].

EVIDENCE:
{evidence_text}
"""

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response_msg = model_obj.generate([{"role": "user", "content": prompt}])
            raw_text = response_msg.content
            return extract_json_from_response(raw_text)
        except Exception as e:
            print(f"      ⚠️  Attempt {attempt}/{RETRY_LIMIT} failed: {e}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY * attempt)

    print(f"      ❌ All {RETRY_LIMIT} attempts failed. Skipping.")
    return []


def get_top_k_event_types(trigger_text: str, sentence_context: str, type_names: List[str], type_embeddings: np.ndarray, sim_model: SentenceTransformer, k: int = 5) -> List[str]:
    # Formulate a query by embedding the trigger with context
    query = f"Trigger '{trigger_text}' in context: {sentence_context}"
    query_emb = sim_model.encode(query)
    
    similarities = []
    for i, t_emb in enumerate(type_embeddings):
        sim = cosine_similarity(query_emb, t_emb)
        similarities.append((sim, type_names[i]))
        
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [name for _, name in similarities[:k]]


def main():
    docs_dir = "eipst_docs"
    if not os.path.isdir(docs_dir):
        print(f"❌ Error: Directory '{docs_dir}' not found.")
        exit(1)
        
    print("--- STARTING HYBRID PIPELINE (GLEN TRIGGER DETECTION + SIMILARITY RANKING + LLM EXTRACTION) ---")

    # Step 1: Read all files and split into sentences
    sentences = []
    file_sentences_map = {} # Track which sentence belongs to which file (optional, but good for tracking)
    
    for filename in os.listdir(docs_dir):
        if not filename.endswith(".txt"):
            continue
            
        filepath = os.path.join(docs_dir, filename)
        try:
            with open(filepath, "r") as f:
                text = f.read()
                
            raw_sentences = text.split('.')
            for s in raw_sentences:
                cleaned = s.strip().replace('\n', ' ')
                if cleaned:
                    if not cleaned.endswith(('.', '?', '!')):
                        cleaned += '.'
                    sentences.append(cleaned)
        except Exception as e:
            print(f"⚠️ Could not read {filepath}: {e}")

    if not sentences:
        print("❌ Error: No sentences extracted from the documents.")
        exit(1)
        
    print(f"✅ Loaded {len(sentences)} sentences from {docs_dir}/")

    # Step 2: Trigger Detection (using GLEN checkpoints)
    ckpt_path = "GLEN/ckpts"
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: GLEN checkpoint not found at {ckpt_path}. Please download it.")
        exit(1)
        
    print("🔍 Running GLEN Trigger Identification...")
    # Change working directory to GLEN temporarily so paths in their code resolve correctly
    original_cwd = os.getcwd()
    os.chdir("GLEN")
    from custom_trigger_detection import detect_triggers
    try:
        triggers_per_sentence = detect_triggers(sentences, "ckpts")
    finally:
        os.chdir(original_cwd)
    
    # Step 3: Embed Event Types for Similarity Calculation
    print("🧠 Loading Sentence Transformer for Event Type Ranking...")
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    event_type_names = list(event_schema.keys())
    # Represent event types by their name and description
    event_type_descriptions = [f"{name}: {event_schema[name]['description']}" for name in event_type_names]
    print("🔄 Embedding Event Types database...")
    type_embeddings = sim_model.encode(event_type_descriptions)
    
    # Process each sentence where triggers were found
    model = GeminiModel()
    all_events = []
    
    for item in triggers_per_sentence:
        sent = item['sentence']
        triggers = item['triggers']
        
        # Filter triggers by confidence > 0.8 to reduce noise
        high_conf_triggers = [t for t in triggers if t['confidence'] > 0.8]
        
        if not high_conf_triggers:
            continue
            
        print(f"\nProcessing sentence with {len(high_conf_triggers)} high-confidence triggers: {sent[:50]}...")
        
        # Gather top K event types across all triggers in this sentence
        sentence_candidate_types = set()
        for t in high_conf_triggers:
            top_k = get_top_k_event_types(t['text'], sent, event_type_names, type_embeddings, sim_model, k=5)
            sentence_candidate_types.update(top_k)
            
        candidate_list = list(sentence_candidate_types)
        print(f"  -> Candidate Event Types based on similarity: {candidate_list}")
        
        # Build prompt focused ONLY on these candidate event types
        system_prompt = build_system_prompt(candidate_list)
        
        # Extract events with LLM
        print("  -> Calling LLM for final extraction...")
        new_events = call_gemini_direct(model, sent, system_prompt)
        print(f"  -> Extracted {len(new_events)} events from this sentence")
        
        all_events.extend(new_events)

    print(f"\n✅ Total events extracted across all sentences: {len(all_events)}")
    
    # Save the aggregated results
    save_events(all_events)

    print("\n--- EXTRACTION COMPLETE ---")
    print(f"📄 EVENTS.json       — {len(all_events)} events")
    print(f"📄 EVENTS_NARRATIVE.txt — updated")


if __name__ == "__main__":
    main()
