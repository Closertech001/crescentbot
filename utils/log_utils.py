import os
import json
from datetime import datetime

def log_query(user_input, score, timestamp=None, feedback=None):
    # Ensure logs/ directory exists
    os.makedirs("logs", exist_ok=True)

    log = {
        "time": timestamp or datetime.now().isoformat(),
        "query": user_input,
        "score": score,
        "feedback": feedback
    }

    with open("logs/query_log.txt", "a", encoding="utf-8") as f:
        f.write(json.dumps(log) + "\n")
