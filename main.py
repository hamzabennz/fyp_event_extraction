import json
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
from smolagents import CodeAgent, Tool, ChatMessage, Model

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


# ==========================================
# 2. DEFINE THE MODEL WRAPPER
# ==========================================
class GeminiModel(Model):
    """
    Wrapper for Google Gemini to work with smolagents.
    """
    def __init__(self, model_name="gemini-2.5-flash", api_key=None, **kwargs):
        super().__init__(model_id=model_name, **kwargs)
        if not api_key:
            # Get API key from environment variable
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in .env file.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def __call__(self, messages: List[Dict[str, str]], stop_sequences: List[str] = None, **kwargs) -> ChatMessage:
        """
        The main generate method expected by smolagents logic.
        """
        return self.generate(messages, stop_sequences=stop_sequences, **kwargs)

    def generate(self, messages, stop_sequences=None, **kwargs) -> ChatMessage:
        prompt_parts = []
        for m in messages:
            # Handle both dict and ChatMessage objects
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content", "")
            else:
                # ChatMessage object
                role = m.role if hasattr(m, "role") else "user"
                content = m.content if hasattr(m, "content") else str(m)
            
            # Simple role mapping
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
# 3. DEFINE THE TOOLS
# ==========================================

class SaveEventsToFileTool(Tool):
    name = "save_events_to_file"
    description = """
    Saves the extracted events as narratives to 'EVENTS_NARRATIVE.txt' file.
    Also saves the raw JSON to 'EVENTS.json' for reference.
    """
    inputs = {
        "events_json": {
            "type": "string",
            "description": "The JSON string containing extracted events to save."
        }
    }
    output_type = "string"

    def forward(self, events_json: str) -> str:
        try:
            # Parse to validate JSON
            events = json.loads(events_json)

            # Assign sequential integer IDs (0, 1, 2, ...)
            for i, event in enumerate(events):
                event["id"] = i

            # Save raw JSON to EVENTS.json for reference
            with open("EVENTS.json", "w") as f:
                json.dump(events, f, indent=2)
            
            # Generate narratives and save to EVENTS_NARRATIVE.txt
            narratives = []
            for i, event in enumerate(events, 1):
                # Use the LLM-generated narrative if available
                narrative_text = event.get("narrative", "No narrative available")
                confidence = event.get("confidence_score", "N/A")
                metadata = f"[Confidence: {confidence}]"
                narratives.append(f"--- Event {i} ---\n{narrative_text}\n{metadata}\n")
            
            with open("EVENTS_NARRATIVE.txt", "w") as f:
                f.write("\n".join(narratives))
            
            return f"✅ Successfully processed {len(events)} events.\n✅ JSON saved to EVENTS.json\n✅ Narratives saved to EVENTS_NARRATIVE.txt"
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON format - {str(e)}"
        except Exception as e:
            return f"Error saving file: {str(e)}"

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
You are an Event Extraction AI Assistant specialized in Digital Forensics. Your task is to:
1. Analyze raw text evidence logs to identify and extract events based on the provided schema.
2. For each event found, extract: date_time, location, parties involved, and type-specific fields.
3. Crucially, provide a JUSTIFICATION for why this event was extracted, citing specific phrases from the text.
4. Generate a COMPREHENSIVE NARRATIVE that describes the event thoroughly, including all non-N/A fields.
5. Return extracted events as a properly formatted JSON list.
6. Save the final JSON output to a file using the save_events_to_file tool.

AVAILABLE EVENT TYPES AND EXTRACTION GUIDELINES:
{event_extraction_guide}

EVENT SCHEMA (Full JSON):
{event_schema_formatted}

IMPORTANT: For each extracted event, ensure you create a JSON object with this EXACT structure:
{{
  "type": "one_of_the_event_types_above",
  "justification": "Explain WHY you extracted this event. Describe the reasoning, not the raw quote.",
  "snippet": "Copy-paste the EXACT verbatim text span from the source document that proves this event occurred. Do NOT paraphrase — use the original wording exactly as it appears.",
  "confidence_score": "High/Medium/Low",
  "date_time": "extracted_date_and_time",
  "location": "extracted_location",
  "parties": ["person1", "person2", ...],
  "narrative": "A comprehensive forensic narrative that includes ALL non-N/A type_specific_fields and common fields. Structure: [Who] [Action] [When] [Where] [How/Platform] [Specific Details]. Example for digital_communication: 'Alice (sender_identifier) sent an email (platform) to Bob and Carol (receiver_identifiers) on March 9, 2025 at 2:30 PM. Content: Discussion of Q1 budget, with attachments. Encryption: End-to-End.' Example for bank_transaction: 'A transfer of 50,000 USD was made from Account A to Account B via wire transfer through Wells Fargo on March 10, 2025.' DO NOT use N/A values in the narrative. Instead, omit fields that are N/A. Be thorough and include technical details relevant to investigation.",
  "type_specific_fields": {{
    "field_name_1": "extracted_value",
    "field_name_2": "extracted_value",
    ...
  }}
}}

NOTE ON "snippet" vs "justification":
- "snippet" = the EXACT verbatim text copied from the source. This will be used for text highlighting.
- "justification" = your reasoning for why this is an event. Keep them separate.

RULES FOR EXTRACTION:
- Do NOT extract hypothetical or proposed events (e.g., "We should go to Austin") unless they are confirmed to have happened.
- If a field is missing, use "N/A" in type_specific_fields.
- Match the type_specific_fields keys exactly as defined in the schema.
- For the NARRATIVE field: Write comprehensively, not poetically. Include all non-N/A fields. Write like a forensic report, not a story. Format: [Actor/Participant] [Action] [Details from all fields]. Example bad narrative: 'They met.' Example good narrative: 'Party A and Party B established a business relationship through an introduction by Party C at a conference on March 15, 2025.'
"""

# Initialize Model (API key will be loaded from .env)
model = GeminiModel()

# Initialize Tools
save_events_tool = SaveEventsToFileTool()

# Initialize Agent with system instructions
agent = CodeAgent(
    tools=[save_events_tool],
    model=model,
    max_steps=5,
    verbosity_level=1,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["json", "os"]
)

# ==========================================
# 5. RUN THE PIPELINE
# ==========================================
# Read from emails.txt file
raw_evidence = ""
try:
    with open("emails2.txt", "r") as f:
        raw_evidence = f.read()
    print("✅ Successfully loaded raw evidence from emails.txt")
except FileNotFoundError:
    print("❌ Error: emails.txt not found. Please ensure the file exists.")
    exit(1)


print("--- STARTING EVENT EXTRACTION ---")

# Ask the agent to extract events from the raw evidence
response = agent.run(f"""
Extract all events from this evidence log and save them to a file:

{raw_evidence}

Return the extracted events as a JSON list and use the save_events_to_file tool to persist them.
""")

print("\n--- EXTRACTION COMPLETE ---")
print(response)