from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .text_utils import extract_urls

SUSPICIOUS_KEYWORDS = {
    "urgent",
    "verify",
    "password",
    "login",
    "click",
    "bank",
    "suspend",
    "locked",
    "confirm",
    "invoice",
    "payment",
    "gift",
    "refund",
    "security",
    "credentials",
    "update",
    "immediately",
    "action required",
    "enable macros",
    "limited time",
    "wire transfer",
    "crypto",
    "otp",
}

SHORTENER_HINTS = ("bit.ly", "tinyurl", "t.co", "goo.gl", "shorturl")


@dataclass
class RuleScanResult:
    suspicious_keywords: List[str]
    urls: List[str]
    heuristic_risk_score: int
    signal_details: List[str]



def score_to_risk_level(score: int) -> str:
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"



def scan_text(text: str) -> RuleScanResult:
    lowered = (text or "").lower()
    matched = sorted([kw for kw in SUSPICIOUS_KEYWORDS if kw in lowered])
    urls = extract_urls(text)
    signals: List[str] = []
    score = 0

    if matched:
        score += min(len(matched) * 7, 42)
        signals.append(f"Matched suspicious keywords: {', '.join(matched[:6])}")
    if urls:
        score += min(len(urls) * 14, 28)
        signals.append(f"Detected {len(urls)} URL(s)")
    if any(token in lowered for token in ["password", "bank", "credentials", "otp"]):
        score += 12
        signals.append("Requests or references sensitive credentials")
    if "enable macros" in lowered:
        score += 18
        signals.append("Contains a macro-enablement instruction")
    if any(hint in lowered for hint in SHORTENER_HINTS):
        score += 12
        signals.append("Contains a shortened URL")
    if "http://" in lowered:
        score += 8
        signals.append("Uses a non-secure HTTP link")
    if not signals:
        signals.append("No strong rule-based phishing signals were detected")

    score = min(score, 100)
    return RuleScanResult(
        suspicious_keywords=matched,
        urls=urls,
        heuristic_risk_score=score,
        signal_details=signals,
    )
