import re

def detect_tone(text):
    """
    Detects the emotional tone of the user's input.

    Args:
        text (str): The input string from the user.

    Returns:
        str: Detected tone - one of ['polite', 'urgent', 'confused', 'angry', 'emphatic', 'neutral']
    """
    text = text.lower()

    # Polite tone
    if any(word in text for word in ["pls", "please", "hi", "hello", "thank", "good morning", "good afternoon", "good evening"]):
        return "polite"

    # Urgent tone
    if any(word in text for word in ["urgent", "now", "quick", "fast", "immediately", "asap"]):
        return "urgent"

    # Confused tone
    if any(word in text for word in ["why", "what", "how", "when", "confused", "help", "explain", "not sure", "don't understand"]):
        return "confused"

    # Angry tone
    if any(word in text for word in ["angry", "nonsense", "rubbish", "dumb", "idiot", "annoyed", "useless", "frustrated", "mad"]):
        return "angry"

    # Emphatic tone (multiple exclamation/question marks)
    if re.search(r"[!?]{2,}", text):
        return "emphatic"

    # Neutral fallback
    return "neutral"


def get_tone_response(response, tone):
    """
    Adjusts the chatbot's response based on detected user tone.

    Args:
        response (str): The original chatbot response.
        tone (str): Detected tone from user input.

    Returns:
        str: Modified chatbot response.
    """
    if tone == "angry":
        return "I'm really sorry if you're feeling frustrated. " + response
    elif tone == "confused":
        return "Let me try to make that clearer for you. " + response
    elif tone == "polite":
        return "Thank you for your politeness! " + response
    elif tone == "urgent":
        return "I'll address that right away. " + response
    elif tone == "emphatic":
