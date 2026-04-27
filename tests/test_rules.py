from src.rules import scan_text, score_to_risk_level


def test_scan_text_detects_keyword_and_url():
    result = scan_text('Urgent: verify your account here https://example.com/login')
    assert 'urgent' in result.suspicious_keywords
    assert result.urls
    assert result.heuristic_risk_score > 0
    assert result.signal_details


def test_risk_level_boundaries():
    assert score_to_risk_level(20) == 'Low'
    assert score_to_risk_level(45) == 'Medium'
    assert score_to_risk_level(80) == 'High'
