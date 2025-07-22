import os
from datetime import datetime


def log_query(query: str, score: float, log_filename: str = "query_log.txt") -> None:
    """
    Logs a query and its similarity score to a log file.

    Args:
        query (str): User query text.
        score (float): Similarity score of top match.
        log_filename (str): Filename to write the log to (default: query_log.txt).
    """
    try:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_filename)

        with open(log_path, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} | Score: {score:.3f} | Query: {query}\n")
    except Exception as e:
        # Optional: print or silently ignore logging errors in production
        print(f"[Log Error] Could not log query: {e}")
