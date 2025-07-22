# ✅ utils/search.py
import openai
from utils.rewrite import rewrite_with_tone

def fallback_to_gpt_if_needed(user_input):
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for Crescent University."},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7
        )
        answer = completion.choices[0].message["content"]
        return rewrite_with_tone(user_input, answer)
    except Exception as e:
        return "Sorry, I'm having trouble reaching my brain right now. Please try again later."
