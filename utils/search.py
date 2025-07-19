import torch
from sentence_transformers import util
from utils.embedding import load_model
from utils.preprocess import preprocess_text, extract_prefix, DEPARTMENT_MAP
import openai
import random

def fallback_openai(user_input, context_qa=None):
    system_prompt = (
        "You are a helpful assistant specialized in Crescent University information. "
        "If you don't know the answer, say so politely and suggest checking university resources."
    )
    messages = [{"role": "system", "content": system_prompt}]

    if context_qa:
        context_text = f"Q: {context_qa['question']}\nA: {context_qa['answer']}\n\n"
        user_message = context_text + "Answer this question: " + user_input
    else:
        user_message = user_input

    messages.append({"role": "user", "content": user_message})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message["content"].strip()
    except Exception:
        return "Sorry, I couldn't reach the server. Try again later."

def find_response(user_input, dataset, embeddings, threshold=0.4):
    model = load_model()
    user_input_clean = preprocess_text(user_input)

    greetings = ["hi", "hello", "hey", "how are you", "what's up"]
    if user_input_clean.lower() in greetings:
        return random.choice([
            "Hello! 👋", "Hi there!", "Greetings!", "How can I help you today?"
        ]), None, 1.0, []

    user_embedding = model.encode(user_input_clean, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(user_embedding, embeddings)[0]
    top_scores, top_indices = torch.topk(cos_scores, k=5)

    top_score = top_scores[0].item()
    top_index = top_indices[0].item()

    if top_score < threshold:
        context_qa = {
            "question": dataset.iloc[top_index]["question"],
            "answer": dataset.iloc[top_index]["answer"]
        }
        return fallback_openai(user_input, context_qa), None, top_score, []

    response = dataset.iloc[top_index]["answer"]
    question = dataset.iloc[top_index]["question"]
    related_questions = [dataset.iloc[i.item()]["question"] for i in top_indices[1:]]

    match = extract_prefix(question)
    department = DEPARTMENT_MAP.get(match, "Unknown") if match else None

    return response, department, top_score, related_questions
