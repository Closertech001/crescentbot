# utils/memory.py

import datetime

def store_context_from_query(user_query, context_store, response):
    """
    Store the user query and corresponding response into short-term memory.
    """
    if "history" not in context_store:
        context_store["history"] = []

    context_store["history"].append({
        "timestamp": datetime.datetime.now().isoformat(),
        "query": user_query.strip(),
        "response": response.strip()
    })

def enrich_query_with_context(user_query, context_store, max_turns=3):
    """
    Add context from previous interactions to the current user query.
    """
    history = context_store.get("history", [])[-max_turns:]
    context = " ".join([f"User: {entry['query']} Bot: {entry['response']}" for entry in history])
    
    if context:
        enriched_query = f"{context} Now the user says: {user_query}"
    else:
        enriched_query = user_query

    return enriched_query

def get_last_context(context_store):
    """
    Retrieve the last question-response pair from memory.
    """
    history = context_store.get("history", [])
    return history[-1] if history else None

def clear_memory(context_store):
    """
    Clears all stored short-term memory.
    """
    context_store["history"] = []
