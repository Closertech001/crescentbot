# utils/tone.py

import re

def detect_tone(text):
    text = text.lower()

    if any(word in text for word in ["pls", "please", "hi", "hello", "thank"]):
        return "polite"
    if any(word in text for word in ["urgent", "now", "quick", "fast"]):
        return "urgent"
    if any(word in text for word in ["why", "what", "how", "when", "confused", "help", "explain"]):
        return "confused"
    if any(word in text for word in ["angry", "nonsense", "rubbish", "dumb", "idiot", "annoyed"]):
        return "angry"
    if any(re.search(r"[!?]{2,}", text)):
        return "emphatic"
    return "neutral"
