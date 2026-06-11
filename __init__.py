# core/__init__.py

from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import json
import logging

logger = logging.getLogger(__name__)

REFRESH_FILE = Path("data/last_refresh.json")


def get_last_refresh():
    if not REFRESH_FILE.exists():
        return None

    try:
        with open(REFRESH_FILE, "r") as f:
            data = json.load(f)

        return datetime.fromisoformat(data["last_refresh"])

    except Exception as e:
        logger.error(f"Failed reading refresh file: {e}")
        return None


def save_refresh_time():
    REFRESH_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(REFRESH_FILE, "w") as f:
        json.dump(
            {"last_refresh": datetime.utcnow().isoformat()},
            f,
            indent=2,
        )


def run_command(name, cmd):
    logger.info(f"Starting {name}")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"{name} failed")

    logger.info(f"{name} completed")


def run_refresh_pipeline():

    run_command(
        "Crawler",
        [
            "uv",
            "run",
            "python",
            "crawler/crawler.py",
            "https://specs.tomtomgroup.com/index.php#ttom",
            "--depth", "5",
            "--threads", "10",
            "--max-pages", "1000",
        ]
    )

    run_command(
        "Scraper",
        [
            "uv",
            "run",
            "python",
            "crawler/scraper.py",
            "crawler/site_map.json",
            "--depth", "2",
            "--chunk-size", "500",
            "--threads", "10",
        ]
    )

    run_command(
        "Ingest",
        [
            "uv",
            "run",
            "python",
            "scripts/ingest_data.py",
            "--data",
            "data/glossary_data.jsonl",
        ]
    )

    save_refresh_time()


def should_refresh():

    last = get_last_refresh()

    if last is None:
        return True

    return datetime.utcnow() - last > timedelta(days=3)