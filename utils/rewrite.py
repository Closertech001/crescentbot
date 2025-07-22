# utils/rewrite.py

from utils.tone import detect_tone

def rewrite_with_tone(user_input, response):
    tone = detect_tone(user_input)

    if tone == "polite":
        return "Sure! 😊 " + response
    elif tone == "urgent":
        return "Got it — here's the information you need right away:\n\n" + response
    elif tone == "confused":
        return "No worries, let me explain clearly:\n\n" + response
    elif tone == "angry":
        return "I'm here to help — let's sort this out calmly:\n\n" + response
    elif tone == "emphatic":
        return "Absolutely! Here's everything you need:\n\n" + response
    else:
        return "Here's what I found for you:\n\n" + response
