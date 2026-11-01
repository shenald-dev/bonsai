"""
Tests for Universal Multi-Archetype Teacher Engine.
"""

import json
from bonsai.core.ingest import DocumentBlock
from bonsai.core.teacher import TeacherEngine, CultivatedPair


def test_teacher_goal_invoice_synthesis():
    teacher = TeacherEngine()
    blocks = [DocumentBlock(source="user_goal", content="Build an AI that extracts strict JSON invoices from receipts", metadata={"type": "goal_prompt"})]
    scenarios = teacher.extract_base_scenarios(blocks)

    assert len(scenarios) >= 4
    for p in scenarios:
        assert isinstance(p, CultivatedPair)
        assert "invoice" in p.instruction.lower() or "receipt" in p.instruction.lower()
        # Verify valid JSON output
        parsed = json.loads(p.output_text)
        assert "invoice_number" in parsed
        assert "total_amount" in parsed


def test_teacher_goal_support_triage_synthesis():
    teacher = TeacherEngine()
    blocks = [DocumentBlock(source="user_goal", content="Classify customer support tickets by urgency and sentiment", metadata={"type": "goal_prompt"})]
    scenarios = teacher.extract_base_scenarios(blocks)

    assert len(scenarios) >= 4
    for p in scenarios:
        parsed = json.loads(p.output_text)
        assert "category" in parsed
        assert "urgency" in parsed
        assert "sentiment" in parsed


def test_teacher_goal_sql_synthesis():
    teacher = TeacherEngine()
    blocks = [DocumentBlock(source="user_goal", content="Convert natural language questions into SQL database queries", metadata={"type": "goal_prompt"})]
    scenarios = teacher.extract_base_scenarios(blocks)

    assert len(scenarios) >= 4
    for p in scenarios:
        parsed = json.loads(p.output_text)
        assert "sql" in parsed
        assert "SELECT" in parsed["sql"] or "DELETE" in parsed["sql"]


def test_teacher_csv_tabular_extraction():
    teacher = TeacherEngine()
    csv_content = """id,name,role,department
1,Alice,Engineer,Platform
2,Bob,Manager,DevOps
3,Charlie,Analyst,Data
"""
    blocks = [DocumentBlock(source="staff.csv", content=csv_content)]
    scenarios = teacher.extract_base_scenarios(blocks)

    assert len(scenarios) == 3
    for p in scenarios:
        parsed = json.loads(p.output_text)
        assert "name" in parsed
        assert "role" in parsed


def test_teacher_code_function_extraction():
    teacher = TeacherEngine()
    code_content = '''
def calculate_p99_latency(samples: list) -> float:
    """Calculate the 99th percentile response time from a list of latencies."""
    samples.sort()
    idx = int(len(samples) * 0.99)
    return samples[idx]

def validate_token_header(token: str) -> bool:
    """Check if bearer auth token has standard length and prefix."""
    return token.startswith("Bearer ") and len(token) > 20
'''
    blocks = [DocumentBlock(source="metrics.py", content=code_content)]
    scenarios = teacher.extract_base_scenarios(blocks)

    assert len(scenarios) == 2
    fn_names = [p.metadata.get("function") for p in scenarios]
    assert "calculate_p99_latency" in fn_names
    assert "validate_token_header" in fn_names


def test_teacher_preformatted_alpaca_and_messages():
    teacher = TeacherEngine()

    alpaca_block = DocumentBlock(source="data.json", content=json.dumps([
        {"instruction": "Add numbers", "input": "2 + 2", "output": "4"}
    ]))
    chat_block = DocumentBlock(source="chat.json", content=json.dumps([
        {"messages": [{"role": "user", "content": "Ping"}, {"role": "assistant", "content": "Pong"}]}
    ]))

    scenarios1 = teacher.extract_base_scenarios([alpaca_block])
    assert len(scenarios1) == 1
    assert scenarios1[0].instruction == "Add numbers"
    assert scenarios1[0].output_text == "4"

    scenarios2 = teacher.extract_base_scenarios([chat_block])
    assert len(scenarios2) == 1
    assert scenarios2[0].input_text == "Ping"
    assert scenarios2[0].output_text == "Pong"


def test_teacher_domain_adaptive_augmentation():
    teacher = TeacherEngine()
    base_pair = CultivatedPair(
        instruction="Extract invoice numbers",
        input_text="INV-100",
        output_text='{"inv": "INV-100"}'
    )
    augmented = teacher.augment_training_scenarios([base_pair], target_count=8)

    assert len(augmented) == 8
    # All augmented pairs should retain the core target output
    for p in augmented:
        assert p.output_text == '{"inv": "INV-100"}'
        assert "Extract invoice numbers" in p.instruction
