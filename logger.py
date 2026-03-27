import json
import os
from datetime import datetime

LOG_FILE = os.path.join(os.path.dirname(__file__), "codi.log")

def log(event: str, data: dict = None):
    if data is None:
        data = {}
    entry = {
        "ts": datetime.now().isoformat(),
        "event": event,
        **data
    }
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # fail silently if log file isn't writable