"""
Tests for Generalized Multi-Metric Proof Machine Evaluator.
"""

from bonsai.evaluators.exam import BonsaiExam


def test_evaluate_json_payload():
    exam = BonsaiExam(base_model_name="mock", adapter_path="mock")

    expected = '{"severity": "CRITICAL", "service": "auth-api", "impact_score": 95}'
    exact_output = '{"severity": "CRITICAL", "service": "auth-api", "impact_score": 95}'
    markdown_output = '```json\n{"severity": "CRITICAL", "service": "auth-api", "impact_score": 95}\n```'
    partial_output = '{"severity": "CRITICAL", "service": "wrong-service"}'
    invalid_output = 'Not a json response'

    valid1, score1 = exam._evaluate_payload(exact_output, expected)
    assert valid1 is True
    assert score1 == 100.0

    valid2, score2 = exam._evaluate_payload(markdown_output, expected)
    assert valid2 is True
    assert score2 == 100.0

    valid3, score3 = exam._evaluate_payload(partial_output, expected)
    assert valid3 is True
    assert 0.0 < score3 < 100.0

    valid4, score4 = exam._evaluate_payload(invalid_output, expected)
    assert valid4 is False
    assert score4 == 0.0


def test_evaluate_freeform_text():
    exam = BonsaiExam(base_model_name="mock", adapter_path="mock")

    expected = "def add(a, b): return a + b"
    exact_out = "def add(a, b): return a + b"
    partial_out = "def add(a, b):\n    # add numbers\n    return a + b"
    mismatched_out = "class Multiply:\n    pass"

    valid1, score1 = exam._evaluate_payload(exact_out, expected)
    assert valid1 is True
    assert score1 == 100.0

    valid2, score2 = exam._evaluate_payload(partial_out, expected)
    assert valid2 is True
    assert score2 > 60.0

    valid3, score3 = exam._evaluate_payload(mismatched_out, expected)
    assert valid3 is False
    assert score3 < 30.0
