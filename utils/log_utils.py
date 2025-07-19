import os
from datetime import datetime

def log_query(query, score):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "query_log.txt"), "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} | Score: {score:.3f} | Query: {query}\n")
