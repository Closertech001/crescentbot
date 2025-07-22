import os
from datetime import datetime

DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_FILE = "query_log.txt"

def log_query(query: str, score: float, log_filename: str = DEFAULT_LOG_FILE, log_dir: str = DEFAULT_LOG_DIR, verbose: bool = False) -> None:
    """
    Logs a query and its similarity score to a timestamped log file.

    Args:
        query (str): The user query.
        score (float): The top similarity score.
        log_filename (str): File to write log to.
        log_dir (str): Directory to store logs.
        verbose (bool): Print to console if True.
    """
    try:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_filename)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"{timestamp} | Score: {score:.3f} | Query: {query}\n"

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_line)

        if verbose:
            print("[Logged]", log_line.strip())

    except Exception as e:
        print(f"[Log Error] Could not log query: {e}")
