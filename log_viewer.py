import time
import json
import os
from typing import List
from rich.live import Live
from rich.table import Table
from rich.console import Console

LOG_FILE = os.path.join(os.path.dirname(__file__), "codi.log")

def read_logs(tail: int = 20) -> List[dict]:
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
        parsed = []
        for line in lines:
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return parsed[-tail:]
    except Exception:
        return []

def generate_table() -> Table:
    table = Table(
        title="CODI Agent Live Telemetry (Press Ctrl+C to exit)",
        show_header=True,
        header_style="bold magenta"
    )
    table.add_column("Timestamp", style="dim", width=25)
    table.add_column("Event", style="cyan", width=22)
    table.add_column("Details", style="green")

    logs = read_logs(30)
    for entry in logs:
        ts = entry.get("ts", "")
        event = entry.get("event", "")
        details_dict = {k: v for k, v in entry.items() if k not in ["ts", "event"]}
        details_str = str(details_dict)
        if len(details_str) > 100:
            details_str = details_str[:97] + "..."

        is_error = (
            "error" in event.lower()
            or details_dict.get("status") in ("error", "rejected")
            or "BLOCKED" in str(details_dict)
        )
        color = "red" if is_error else "green"
        table.add_row(ts, event, f"[{color}]{details_str}[/{color}]")

    return table

def run_viewer():
    console = Console()
    console.clear()
    try:
        with Live(generate_table(), refresh_per_second=2, screen=True) as live:
            while True:
                time.sleep(1)
                live.update(generate_table())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    run_viewer()