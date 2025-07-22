from textblob import TextBlob

def detect_tone(text):
    """Simple tone detection using sentiment polarity."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.3:
        return "positive"
    elif polarity < -0.3:
        return "negative"
    else:
        return "neutral"

def get_tone_response(response, tone):
    """Modify response based on user tone for empathy."""
    if tone == "negative":
        return "I'm here to help. " + response
    elif tone == "positive":
        return "Great! " + response
    return response
