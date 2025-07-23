# utils/search.py

import openai

def fallback_to_gpt_if_needed(user_input, similarity_score, matched_answer, threshold=0.75):
    if similarity_score >= threshold:
        return matched_answer

    prompt = (
        f"The user asked: \"{user_input}\"\n"
        "This question could not be answered from the database.\n"
        "Please provide a helpful and friendly answer related to Crescent University."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Sorry, I couldn't retrieve a full answer right now. Please try again."
