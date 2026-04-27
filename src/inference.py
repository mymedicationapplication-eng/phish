from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

import joblib
import numpy as np

from .config import LABEL_MAP, METRICS_PATH, MODEL_PATH
from .rules import scan_text, score_to_risk_level
from .text_utils import normalize_text


class ModelNotFoundError(FileNotFoundError):
    pass


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise ModelNotFoundError("Model artifact was not found. Please run 'python train_model.py' first.")
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_metrics(path: Path = METRICS_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def clear_caches() -> None:
    load_model.cache_clear()
    load_metrics.cache_clear()


def _extract_top_terms(model, cleaned: str, top_n: int = 6) -> List[str]:
    try:
        vectorizer = model.named_steps['vectorizer']
        classifier = model.named_steps['classifier']
        matrix = vectorizer.transform([cleaned])
        if matrix.nnz == 0:
            return []
        feature_names = np.asarray(vectorizer.get_feature_names_out())
        coefs = classifier.coef_[0]
        row = matrix.tocoo()
        contributions = []
        for idx, value in zip(row.col, row.data):
            score = float(value * coefs[idx])
            if score > 0:
                contributions.append((score, feature_names[idx]))
        contributions.sort(reverse=True)
        return [term for _, term in contributions[:top_n]]
    except Exception:
        return []


def build_recommendation(predicted_name: str, risk_level: str, urls: List[str], signal_count: int) -> str:
    if predicted_name == 'Phishing' or risk_level == 'High':
        if urls:
            return (
                'Do not click the detected links. Verify the sender through an independent channel, '
                'do not share credentials, and report the message to your instructor, administrator, or security team.'
            )
        return (
            'Treat this message as suspicious. Avoid sharing credentials or financial details until the sender '
            'and the request are validated independently.'
        )
    if risk_level == 'Medium' or signal_count >= 2:
        return 'Review the sender identity, links, urgency cues, and language carefully before taking any action.'
    return 'The message appears relatively safe, but normal verification is still recommended before sharing sensitive information.'


def _build_result(text: str, model) -> Dict[str, Any]:
    cleaned = normalize_text(text)
    if not cleaned:
        raise ValueError('Input text is empty after normalization.')

    probabilities = model.predict_proba([cleaned])[0]
    predicted_label = int(probabilities.argmax())
    confidence = float(probabilities[predicted_label])
    phishing_probability = float(probabilities[1])
    legitimate_probability = float(probabilities[0])
    rules = scan_text(text)
    risk_score = rules.heuristic_risk_score
    risk_level = score_to_risk_level(risk_score)
    ml_top_terms = _extract_top_terms(model, cleaned)

    explanation_parts = []
    if rules.suspicious_keywords:
        explanation_parts.append(f"matched suspicious keywords such as {', '.join(rules.suspicious_keywords[:5])}")
    if rules.urls:
        explanation_parts.append(f'detected {len(rules.urls)} URL(s)')
    if ml_top_terms:
        explanation_parts.append(f"included influential model terms such as {', '.join(ml_top_terms[:4])}")
    if not explanation_parts:
        explanation_parts.append('did not match strong rule-based phishing indicators')

    recommendation = build_recommendation(LABEL_MAP[predicted_label], risk_level, rules.urls, len(rules.signal_details))
    return {
        'normalized_text': cleaned,
        'predicted_label': predicted_label,
        'predicted_name': LABEL_MAP[predicted_label],
        'confidence': confidence,
        'phishing_probability': phishing_probability,
        'legitimate_probability': legitimate_probability,
        'suspicious_keywords': rules.suspicious_keywords,
        'detected_urls': rules.urls,
        'heuristic_risk_score': risk_score,
        'risk_level': risk_level,
        'signal_count': len(rules.signal_details),
        'signal_details': rules.signal_details,
        'ml_top_terms': ml_top_terms,
        'recommendation': recommendation,
        'explanation': f"The message was classified as {LABEL_MAP[predicted_label]} because it {', and '.join(explanation_parts)}.",
    }


def predict_text(text: str) -> Dict[str, Any]:
    model = load_model()
    return _build_result(text, model)


def predict_many(texts: Iterable[str]) -> List[Dict[str, Any]]:
    model = load_model()
    results: List[Dict[str, Any]] = []
    for text in texts:
        try:
            results.append(_build_result(text, model))
        except ValueError:
            continue
    return results
