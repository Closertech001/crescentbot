from datetime import datetime

def log_query(user_input, score, timestamp=None, feedback=None):
    log = {
        "time": timestamp or datetime.now().isoformat(),
        "query": user_input,
        "score": score,
        "feedback": feedback
    }
    with open("logs/query_log.txt", "a", encoding="utf-8") as f:
        f.write(str(log) + "\n")
