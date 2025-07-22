# ✅ utils/greetings.py
def detect_greeting(text):
    greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    return any(greet in text.lower() for greet in greetings)

def generate_greeting_response():
    return "Hello! 👋 How can I assist you about Crescent University today?"

def is_small_talk(text):
    text = text.lower()
    return any(phrase in text for phrase in ["how are you", "what's up", "are you real", "who made you"])

def generate_small_talk_response():
    responses = [
        "I'm just a friendly chatbot made to help you 😊",
        "I'm doing great! Hope you're too!",
        "Built by amazing minds to assist Crescent University students.",
        "I don't sleep, so I'm always here for you! 😄"
    ]
    return random.choice(responses)
