import inspect
import json
import os
import queue
import threading
from datetime import datetime

LOG_FILE = os.path.join(os.path.dirname(__file__), "codi.log")
_CALLER_DEBUG = os.getenv("CODI_CALLER_DEBUG", "0") == "1"
_LOG_QUEUE: "queue.Queue[dict]" = queue.Queue()
_LOG_WORKER_STARTED = False
_LOG_WORKER_LOCK = threading.Lock()


def _log_worker() -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        while True:
            entry = _LOG_QUEUE.get()
            if entry is None:
                break
            f.write(json.dumps(entry, ensure_ascii=True, default=str) + "\n")
            f.flush()


def _start_log_worker() -> None:
    global _LOG_WORKER_STARTED
    with _LOG_WORKER_LOCK:
        if _LOG_WORKER_STARTED:
            return
        worker = threading.Thread(target=_log_worker, daemon=True, name="codi-log-worker")
        worker.start()
        _LOG_WORKER_STARTED = True


def log(event: str, data: dict = None):
    if data is None:
        data = {}
    caller = _caller_info() if _CALLER_DEBUG else {}
    entry = {
        "ts": datetime.now().isoformat(),
        "event": event,
        "pid": os.getpid(),
        "thread": threading.current_thread().name,
        **caller,
        **data,
    }
    try:
        _LOG_QUEUE.put_nowait(entry)
    except Exception:
        pass  # fail silently if the queue is full or unavailable


_start_log_worker()


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
