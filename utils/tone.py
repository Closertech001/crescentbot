import re

def detect_tone(text):
    text = text.lower()
    if any(word in text for word in ["pls", "please", "hi", "hello", "thank", "good morning", "good afternoon", "good evening"]):
        return "polite"
    if any(word in text for word in ["urgent", "now", "quick", "fast", "immediately", "asap"]):
        return "urgent"
    if any(word in text for word in ["why", "what", "how", "when", "confused", "help", "explain", "not sure", "don't understand"]):
        return "confused"
    if any(word in text for word in ["angry", "nonsense", "rubbish", "dumb", "idiot", "annoyed", "useless", "frustrated", "mad"]):
        return "angry"
    if re.search(r"[!?]{2,}", text):
        return "emphatic"
    return "neutral"

def get_tone_response(response, tone):
    if tone == "angry":
        return "I'm really sorry if you're feeling frustrated. " + response
    elif tone == "confused":
        return "Let me try to make that clearer for you. " + response
    elif tone == "polite":
        return "Thank you for your politeness! " + response
    elif tone == "urgent":
        return "I'll address that right away. " + response
    elif tone == "emphatic":
        return "No worries, I got you! " + response
    return response
