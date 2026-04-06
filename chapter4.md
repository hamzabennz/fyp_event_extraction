Chapter 4: Phase 2 — Understanding Events and Building the Investigation Story
4.1 Introduction

In the first phase of our project, we focused on gathering all the necessary information from different sources. However, having a large amount of data is only the beginning. For an investigator, the real challenge is reading through thousands of emails, messages, and documents to find the "hidden story."

Phase 2 of our system is designed to solve this problem. It takes all the raw text we found and turns it into a clear, organized map of events. The system identifies important actions, groups them into themes, and allows the investigator to check everything for accuracy. The goal is to move from a pile of documents to a clear understanding of what happened, who was involved, and why it matters.
4.2 The Step-by-Step Process

To make the investigation reliable, our system follows a specific sequence. Instead of trying to do everything at once, it breaks the work down into smaller, logical steps. This ensures that every piece of evidence is handled carefully and that a human expert always has the final say. Our backend architecture implements a robust 7-stage pipeline that orchestrates everything from data ingestion to the final visual output, seamlessly integrating multiple specialized Large Language Models.

4.3 Finding Important Events Automatically

The first task of this phase is to look through the documents and find "events." An event is any specific action that might be important to the case, such as a bank transaction, a scheduled meeting, or a suspicious email.

The AI is guided by a predefined forensic schema (containing 16 targeted event types) to look for specific categories, such as:

    Communications: Phone calls or digital messages.

    Money: Financial transactions or illicit exchanges.

    Travel: Trips or movement between locations.

We utilize the Gemini model to process evidence files in batches. For every event the AI finds, it is forced to output a strictly structured record that includes the extracted date and time, location, parties involved, a comprehensive narrative, and a confidence score. Crucially, it must provide a justification that explains why that event is relevant and link it directly back to the original verbatim text snippet from the document. This provenance ensures that the investigator can always verify where the information came from.

4.4 The Expert Review (Human-in-the-Loop)

AI can be very helpful, but it is not perfect. In legal or police work, we cannot afford mistakes. Because of this, our system includes a strict "Review Gate."

After the LLM extracts the events, the pipeline automatically blocks and waits for the human investigator. The investigator uses a web interface to inspect the AI's findings. They can:

    Approve events that are correct and important.

    Deselect or delete false positives.

    Verify the information by checking the highlighted snippet from the original text.

The system will not move to the next step until the human expert has submitted their review, at which point the system permanently discards the rejected events and moves forward with only verified data.

4.5 Seeing the Big Picture: Finding Patterns via Concept Induction

Once the events are verified, the system looks for "patterns." Individual events like a single email might not mean much on their own, but when several events are put together, they tell a bigger story.

To accomplish this, our pipeline integrates LLooM, an advanced concept induction framework framework. We utilize a sequence of four distinct LLM operations:

1. Distillation: Summarizing each verified event into a concise bullet point.
2. Clustering: Using an embedding model to group related event summaries together.
3. Synthesis: Generating a high-level concept or theme name for each cluster (e.g., "Financial and Logistical Coordination").
4. Scoring: Evaluating every single event against every induced concept to assign a relevance score (from 0 to 1).

The system then performs a subsequent finding synthesis step for high-scoring events. It gives each theme an evidentiary "strength score": STRONG, MODERATE, or CIRCUMSTANTIAL based on the volume and quality of the underlying events. This structured synthesis explains the pattern, key actors, timeframe, and significance, helping the investigator focus their time on the most critical parts of the case.

4.6 The Final Result: An Interactive Mindmap

The final output of this phase is an Interactive Mindmap. The system aggregates the verified events, their source snippets, the induced concepts, and the synthesized findings into a self-contained, interactive HTML file that the investigator can open in any web browser.

The mindmap is organized like a tree:

    The Main Case: At the center is the overall investigation.

    Themes (Concepts): Branching out are the major patterns the AI found (like "Financial Fraud" or "Coordinated Travel"), along with their strength ratings (e.g., STRONG).

    Events: Inside each theme are the specific events that prove it, including the date, parties, and relevance score.

    Original Evidence: By clicking on an event, the investigator can see the exact verbatim sentence from the original file, maintaining the unbroken chain of evidence from raw text to high-level theory.

4.7 Conclusion

Phase 2 transforms a confusing "data dump" into a structured and easy-to-read story. By combining the speed of AI with the wisdom of a human expert, the system ensures that investigations are both fast and accurate. The final mindmap serves as a complete guide to the case, allowing an investigator to see the big picture without losing sight of the small details. This significantly reduces the time and effort needed to solve complex cases.