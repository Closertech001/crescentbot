# ✅ utils/tone.py
import re

def detect_tone(text):
    text = text.lower()
    if any(word in text for word in ["pls", "please", "hi", "hello", "thank", "good morning", "good afternoon", "good evening"]):
        return "polite"
    if any(word in text for word in ["urgent", "now", "quick", "fast", "immediately", "asap"]):
        return "urgent"
    if any(word in text for word in ["why", "what", "how", "when", "confused", "help", "not sure", "don't understand"]):
        return "confused"
    if any(word in text for word in ["angry", "nonsense", "rubbish", "dumb", "idiot", "annoyed", "useless", "frustrated", "mad"]):
        return "angry"
    if re.search(r"[!?]{2,}", text):
        return "emphatic"
    return "neutral"
