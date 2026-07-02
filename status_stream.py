from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Callable

from logger import log


class StatusStream:
    def __init__(self, max_lines: int = 8) -> None:
        self.max_lines = max_lines
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._callbacks: list[Callable[[str], None]] = []
        self._lock = Lock()

    def register(self, callback: Callable[[str], None]) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def unregister(self, callback: Callable[[str], None]) -> None:
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def emit(self, prefix: str, message: str) -> None:
        line = f"{prefix} {message}".strip()
        if not line:
            return
        log("status_event", {"prefix": prefix, "message": message})
        with self._lock:
            self._lines.append(line)
            for callback in list(self._callbacks):
                callback(line)

    def snapshot(self) -> list[str]:
        with self._lock:
            return list(self._lines)


_status_stream = StatusStream()


def emit_status(prefix: str, message: str) -> None:
    _status_stream.emit(prefix, message)


def register_status_callback(callback: Callable[[str], None]) -> None:
    _status_stream.register(callback)


def unregister_status_callback(callback: Callable[[str], None]) -> None:
    _status_stream.unregister(callback)


def get_status_snapshot() -> list[str]:
    return _status_stream.snapshot()
