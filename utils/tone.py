# utils/tone.py

import random

# 🎉 Dynamic intro phrases for answers
INTRO_PHRASES = [
    "Here’s what I found for you 😊:",
    "Alright, take a look at this 📘:",
    "This might help 🎓:",
    "Let’s break it down 🔍:",
    "I’ve got something for you 💡:",
    "Here's the info you asked for 📖:",
    "Check this out 👇:",
    "Hope this helps 📚:"
]

# 😕 Fallback responses when nothing is found
NO_MATCH_PHRASES = [
    "😕 I couldn’t find an answer to that. Try rephrasing it?",
    "🤔 I’m not sure about that one. Can you ask differently?",
    "🙈 I didn’t catch that. Could you clarify?",
    "Sorry, I don't have that information right now. Try asking in another way.",
    "Hmm… I couldn’t match that to anything in my knowledge base.",
    "That’s a bit unclear. Could you be more specific?"
]

# ✨ Randomized intro
def get_intro_phrase() -> str:
    return random.choice(INTRO_PHRASES)

# 😢 Randomized fallback
def get_no_match_phrase() -> str:
    return random.choice(NO_MATCH_PHRASES)
