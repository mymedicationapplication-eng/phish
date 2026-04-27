import re
from typing import List

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z0-9\s:/._@-]")
MULTISPACE_PATTERN = re.compile(r"\s+")
EMAIL_PATTERN = re.compile(r"^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$", re.IGNORECASE)


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = NON_ALPHA_PATTERN.sub(" ", text)
    text = MULTISPACE_PATTERN.sub(" ", text)
    return text.lower().strip()


def extract_urls(text: str) -> List[str]:
    return URL_PATTERN.findall(text or "")


def is_valid_email(email: str) -> bool:
    return bool(EMAIL_PATTERN.match((email or "").strip()))
