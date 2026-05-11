import sys
import os
import json

# Add GLEN directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "GLEN"))
from custom_trigger_detection import detect_triggers

def main():
    with open("../input.txt", "r") as f:
        text = f.read()

    # Simple sentence splitting on newlines and periods
    # For a more robust solution, we could use nltk.tokenize.sent_tokenize
    raw_sentences = text.split('.')
    
    sentences = []
    for s in raw_sentences:
        # Strip and replace newlines with space
        cleaned = s.strip().replace('\n', ' ')
        if cleaned:
            # Re-add period that was lost in split if it's not a header-like short string
            if not cleaned.endswith(('.', '?', '!')):
                cleaned += '.'
            sentences.append(cleaned)
    
    # We will pass None for ckpt_path since we don't have the weights downloaded.
    # The script will initialize with random weights.
    ckpt_path = "ckpts" 
    
    print(f"Running trigger detection on {len(sentences)} sentences...")
    
    try:
        triggers = detect_triggers(sentences, ckpt_path)
        
        output_file = "output_triggers.json"
        with open(output_file, "w") as f:
            json.dump(triggers, f, indent=2)
            
        print(f"Successfully processed input and saved results to {output_file}")
        
        # Also print a summary
        for res in triggers:
            if res['triggers']:
                print(f"\nSentence: {res['sentence']}")
                for t in res['triggers']:
                    print(f"  -> Trigger: '{t['text']}' (Confidence: {t['confidence']:.4f})")
                    
    except Exception as e:
        print(f"Error during trigger detection: {e}")

if __name__ == "__main__":
    main()
