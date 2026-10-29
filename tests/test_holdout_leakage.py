"""
Tests for Zero Data Leakage Split & Holdout Verification in Bonsai.
"""

import json
from pathlib import Path
from bonsai.core.ingest import DocumentBlock
from bonsai.core.teacher import TeacherEngine
from bonsai.core.holdout import cultivate_and_split_zero_leakage

SAMPLE_LOGS = """
[2026-10-28 01:00:00] CRITICAL: db-cluster connection pool exhausted on worker-node-01. Max connections reached.
[2026-10-28 02:00:00] FATAL: billing-worker pod terminated by OOM-Killer. Memory exceeded 2048Mi.
[2026-10-28 03:00:00] WARN: redis-cache node-02 observed high eviction rate. Cache hit ratio dropped to 70%.
[2026-10-28 04:00:00] ERROR: payment-gateway webhook timed out after 30000ms. HTTP 504.
[2026-10-28 05:00:00] CRITICAL: auth-api JWT signature validation latency escalated to 4500ms.
[2026-10-28 06:00:00] WARN: order-service message backlog on Kafka topic reached 80000 uncommitted offsets.
"""

def test_zero_data_leakage(tmp_path):
    blocks = [DocumentBlock(source="test_logs.txt", content=SAMPLE_LOGS)]
    teacher = TeacherEngine()

    train_path, exam_path, stats = cultivate_and_split_zero_leakage(
        blocks=blocks,
        output_dir=tmp_path,
        teacher=teacher,
        holdout_ratio=0.20,
        target_train_size=16,
        seed=42
    )

    assert train_path.exists()
    assert exam_path.exists()
    assert stats["overlap_count"] == 0

    # Read both datasets and verify contents
    train_lines = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    exam_lines = [json.loads(line) for line in exam_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(train_lines) >= stats["train_scenarios_count"]
    assert len(exam_lines) == stats["exam_scenarios_count"]

    # Extract user input texts from both datasets
    train_user_inputs = set()
    for item in train_lines:
        for msg in item["messages"]:
            if msg["role"] == "user":
                train_user_inputs.add(msg["content"].strip().lower())

    exam_user_inputs = set()
    for item in exam_lines:
        for msg in item["messages"]:
            if msg["role"] == "user":
                exam_user_inputs.add(msg["content"].strip().lower())

    # Calculate Jaccard similarity between train and exam inputs
    intersection = train_user_inputs.intersection(exam_user_inputs)
    assert len(intersection) == 0, f"Data leakage detected! Overlapping prompts: {intersection}"

    # Verify both sets include the system prompt
    for item in train_lines + exam_lines:
        roles = [m["role"] for m in item["messages"]]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles

def test_deterministic_seed_reproducibility(tmp_path):
    blocks = [DocumentBlock(source="test_logs.txt", content=SAMPLE_LOGS)]
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"

    t1, e1, s1 = cultivate_and_split_zero_leakage(blocks, dir1, seed=123)
    t2, e2, s2 = cultivate_and_split_zero_leakage(blocks, dir2, seed=123)

    assert e1.read_text(encoding="utf-8") == e2.read_text(encoding="utf-8")
    assert s1["exam_scenarios_count"] == s2["exam_scenarios_count"]
