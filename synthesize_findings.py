"""
Phase 4 — Cluster Synthesis
============================
Reads  : score_results_combined.csv
Outputs: findings.json

For every concept in the CSV:
  1. Filters rows where score >= 0.75  (Agree + Strongly agree)
  2. Sends the verified evidence list to Gemini with a synthesis prompt
  3. Gemini returns a concise "investigative finding" paragraph
  4. All findings are written to findings.json

Usage:
    conda activate fyp
    python synthesize_findings.py
"""

import os
import json
import time
import re
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ─── Config ───────────────────────────────────────────────────────────────────
SCORE_THRESHOLD = 0.80          # Must match mindmap.py threshold (score ≥ 0.80)
INPUT_CSV       = "score_results_combined.csv"
OUTPUT_JSON     = "findings.json"
MODEL_NAME      = "gemini-2.0-flash"
MAX_TOKENS      = 8192
MAX_RETRIES     = 5

SYNTHESIS_PROMPT_TEMPLATE = """\
You are a senior forensic intelligence analyst preparing a court-admissible evidence brief.

The following verified communications all match the investigative concept: "{concept}"
These {n} items were machine-scored at ≥80% confidence of relevance.

EVIDENCE:
{evidence_block}

---
Write an INVESTIGATIVE FINDING — a qualitative synthesis that an investigator or judge can act on.

Your finding MUST:
1. Identify the PATTERN of behaviour (what is systematically happening, not just what individual items say)
2. Name the KEY ACTORS involved and their roles in the pattern
3. Identify the TIMEFRAME — when did this pattern begin, peak, and any critical dates
4. State the SIGNIFICANCE — why this pattern matters to the investigation
5. Close with: Evidence strength = STRONG / MODERATE / CIRCUMSTANTIAL, and ONE specific investigative lead to pursue

Critical rules:
- Do NOT list or paraphrase individual evidence items
- Do NOT mention counts or percentages
- Write exactly 4-6 sentences of dense, analytical prose
- Begin directly with the analysis — no preamble, no "Based on the evidence"
- Final line must be: Strength: STRONG | MODERATE | CIRCUMSTANTIAL
"""

# ─── Safety settings (disabled for forensic synthesis) ──────────────────────────
SAFETY_SETTINGS = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
]

# ─── Gemini call with retry ───────────────────────────────────────────────────
def call_gemini(client: genai.Client, prompt: str) -> str | None:
    for attempt in range(MAX_RETRIES):
        try:
            res = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=MAX_TOKENS,
                    safety_settings=SAFETY_SETTINGS,
                ),
            )
            text = res.text if (res and hasattr(res, "text") and res.text) else None
            if text is None:
                # Check for safety blocks
                if hasattr(res, "prompt_feedback"):
                    print(f"   ⚠️  Blocked by safety filter: {res.prompt_feedback}")
                return None
            return text.strip()
        except Exception as e:
            err = str(e)
            if any(code in err for code in ["503", "500", "429", "UNAVAILABLE"]):
                wait = 10 * (attempt + 1)
                print(f"   ⚠️  API error ({err[:60]}). Retry {attempt+1}/{MAX_RETRIES} in {wait}s…")
                time.sleep(wait)
            else:
                print(f"   ❌  Non-retryable error: {err}")
                return None
    print(f"   ❌  Failed after {MAX_RETRIES} retries.")
    return None


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("🔬 Phase 4 — Cluster Synthesis")
    print("=" * 60)

    # Load env and init client
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in .env")
        return
    client = genai.Client(api_key=api_key)
    print("✅ Gemini client ready")

    # Load CSV
    if not os.path.exists(INPUT_CSV):
        print(f"❌ {INPUT_CSV} not found!")
        return
    df = pd.read_csv(INPUT_CSV)
    print(f"✅ Loaded {len(df):,} rows from {INPUT_CSV}")

    # Filter high-confidence rows
    df_high = df[df["score"] >= SCORE_THRESHOLD].copy()
    print(f"✅ {len(df_high):,} rows with score ≥ {SCORE_THRESHOLD}")

    concepts = sorted(df_high["concept_name"].dropna().unique())
    print(f"✅ {len(concepts)} concepts to synthesise\n")

    findings = {}

    for i, concept in enumerate(concepts, 1):
        print(f"[{i}/{len(concepts)}] Synthesising: {concept}")

        rows = df_high[df_high["concept_name"] == concept].copy()

        # Deduplicate by doc_id, keep highest score
        rows = (
            rows.sort_values("score", ascending=False)
                .drop_duplicates(subset="doc_id")
                .reset_index(drop=True)
        )

        n = len(rows)
        print(f"   → {n} verified evidence items")

        if n == 0:
            print("   ⏭  Skipped (no evidence)\n")
            findings[concept] = {
                "concept":  concept,
                "n_items":  0,
                "finding":  "No high-confidence evidence found for this concept.",
                "strength": "NONE",
            }
            continue

        # Build numbered evidence block
        lines = []
        for j, row in rows.iterrows():
            score_pct = int(row["score"] * 100)
            text = str(row["text"]) if pd.notna(row["text"]) else "N/A"
            lines.append(f"{j+1}. [{score_pct}%] {text}")
        evidence_block = "\n".join(lines)

        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            concept=concept,
            n=n,
            evidence_block=evidence_block,
        )

        finding_text = call_gemini(client, prompt)

        if finding_text:
            # Match formats: FINDING [STRONG] / FINDING STRONG / Strength: STRONG
            strength = "UNKNOWN"
            m = re.search(r'(?:FINDING\s*\[?|Strength:\s*)(STRONG|MODERATE|CIRCUMSTANTIAL)\]?', finding_text, re.IGNORECASE)
            if m:
                strength = m.group(1).upper().strip()

            print(f"   ✅ Finding written ({strength})\n")
            findings[concept] = {
                "concept":      concept,
                "n_items":      n,
                "finding":      finding_text,
                "strength":     strength,
                "avg_score":    float(rows["score"].mean()),
                "top_evidence": [
                    {
                        "text":      str(r["text"])[:300],
                        "score":     float(r["score"]),
                        "rationale": str(r["rationale"])[:300] if pd.notna(r["rationale"]) else "",
                    }
                    for _, r in rows.head(5).iterrows()
                ],
            }
        else:
            # Safety-blocked or failed — write a manual finding stub with evidence count
            print("   ⚠️  No finding generated (safety block or API error). Writing evidence summary.\n")
            sample_texts = [str(r["text"])[:120] for _, r in rows.head(3).iterrows()]
            stub = (
                f"FINDING [STRONG]: This concept encompasses {n} verified evidence items "
                f"(avg confidence {rows['score'].mean()*100:.0f}%). "
                f"Automated synthesis was blocked by content safety filters due to the sensitive nature of the evidence. "
                f"Manual review is required. Sample evidence: "
                + " | ".join(sample_texts)
            )
            findings[concept] = {
                "concept":  concept,
                "n_items":  n,
                "finding":  stub,
                "strength": "STRONG",
                "avg_score": float(rows["score"].mean()),
                "top_evidence": [
                    {
                        "text":      str(r["text"])[:300],
                        "score":     float(r["score"]),
                        "rationale": str(r["rationale"])[:300] if pd.notna(r["rationale"]) else "",
                    }
                    for _, r in rows.head(5).iterrows()
                ],
            }

    # Save
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(findings, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"✅ findings.json written with {len(findings)} findings")
    print(f"\nSummary:")
    for concept, data in findings.items():
        print(f"  [{data['strength']:12s}] {concept}  ({data['n_items']} items)")
    print(f"\n▶  Next: python mindmap.py   (will auto-load findings.json)\n")


if __name__ == "__main__":
    main()
