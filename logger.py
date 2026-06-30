import inspect
import json
import os
import threading
from datetime import datetime

LOG_FILE = os.path.join(os.path.dirname(__file__), "codi.log")

def log(event: str, data: dict = None):
    if data is None:
        data = {}
    caller = _caller_info()
    entry = {
        "ts": datetime.now().isoformat(),
        "event": event,
        "pid": os.getpid(),
        "thread": threading.current_thread().name,
        **caller,
        **data
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True, default=str) + "\n")
    except Exception:
        pass  # fail silently if log file isn't writable


def _caller_info() -> dict:
    """Return the first non-logger caller so every event has source metadata."""
    try:
        frame = inspect.currentframe()
        if frame is None:
            return {}
        frame = frame.f_back
        while frame:
            module = inspect.getmodule(frame)
            module_name = module.__name__ if module else ""
            if module_name != __name__:
                code = frame.f_code
                return {
                    "caller_module": module_name or "<unknown>",
                    "caller_function": code.co_name,
                    "caller_file": os.path.abspath(code.co_filename),
                    "caller_line": frame.f_lineno,
                }
            frame = frame.f_back
    except Exception:
        return {}
    return {}
