# core/__init__.py


"""
app/utils/sanitise.py
Extra sanitisation helpers beyond Pydantic validators.
"""
import re
import unicodedata


# Characters that could be used for prompt injection
_INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"forget\s+everything",
    r"you\s+are\s+now",
    r"act\s+as\s+",
    r"jailbreak",
    r"<\s*script",           # basic XSS
    r";\s*drop\s+table",     # SQL injection (just in case)
]
_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS),
    re.IGNORECASE,
)


def is_prompt_injection(text: str) -> bool:
    """Return True if text looks like a prompt-injection attempt."""
    return bool(_INJECTION_RE.search(text))


def normalise_text(text: str) -> str:
    """
    NFC-normalise unicode, collapse whitespace,
    strip leading/trailing space.
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_query(query: str) -> str:
    """Full clean pipeline for user queries."""
    query = normalise_text(query)
    # Remove zero-width chars
    query = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", query)
    return query
