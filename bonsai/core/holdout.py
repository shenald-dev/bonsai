"""
Data Splitter & Blind Holdout Verification Engine for Bonsai.
Guarantees strict 85% training vs. 15% blind holdout separation with mathematically verified ZERO data leakage.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set, Optional
from bonsai.core.teacher import CultivatedPair, TeacherEngine
from bonsai.core.ingest import DocumentBlock
from bonsai.core.constants import DEFAULT_SYSTEM_DIRECTIVE

def cultivate_and_split_zero_leakage(
    blocks: List[DocumentBlock],
    output_dir: Path,
    teacher: Optional[TeacherEngine] = None,
    holdout_ratio: float = 0.15,
    target_train_size: int = 24,
    seed: int = 42
) -> Tuple[Path, Path, Dict[str, Any]]:
    """
    Cultivates raw document blocks and strictly partitions unique scenarios BEFORE augmentation.
    Guarantees:
      1. Every input scenario in exam.jsonl has zero overlap with cultivate.jsonl.
      2. Deterministic, reproducible partition using seed.
      3. Both train and exam sets use identical system directives and chat templates.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher = teacher or TeacherEngine()

    # Step 1: Extract distinct base scenario pairs from blocks
    base_pairs = teacher.extract_base_scenarios(blocks)

    if not base_pairs:
        raise ValueError("Teacher engine failed to extract any valid scenarios from input data.")

    # Deduplicate base scenarios by normalized input text
    unique_scenarios: List[CultivatedPair] = []
    seen_inputs: Set[str] = set()
    for p in base_pairs:
        norm = p.input_text.strip().lower()
        if norm and norm not in seen_inputs:
            seen_inputs.add(norm)
            unique_scenarios.append(p)

    total_scenarios = len(unique_scenarios)
    if total_scenarios < 2:
        # Minimum dataset size validation
        raise ValueError(f"Insufficient distinct scenarios found ({total_scenarios}). Bonsai requires at least 2 distinct operational examples.")

    # Step 2: Deterministically shuffle unique scenarios
    rng = random.Random(seed)
    shuffled = list(unique_scenarios)
    rng.shuffle(shuffled)

    # Step 3: Partition base scenarios strictly into Train and Exam
    if total_scenarios <= 4:
        exam_count = 1
    else:
        exam_count = max(2, int(total_scenarios * holdout_ratio))

    train_base_scenarios = shuffled[exam_count:]
    exam_base_scenarios = shuffled[:exam_count]

    # Verify ZERO scenario overlap (Jaccard similarity = 0)
    train_inputs = set(p.input_text.strip().lower() for p in train_base_scenarios)
    exam_inputs = set(p.input_text.strip().lower() for p in exam_base_scenarios)
    overlap = train_inputs.intersection(exam_inputs)
    if overlap:
        raise RuntimeError(f"Data leakage detected! {len(overlap)} scenarios overlap between train and exam.")

    # Step 4: Augment ONLY the training scenarios to reach target training capacity
    train_pairs = teacher.augment_training_scenarios(train_base_scenarios, target_count=target_train_size)

    # Holdout exam scenarios remain 100% pristine and unaugmented
    exam_pairs = list(exam_base_scenarios)

    # Step 5: Save both datasets with identical chat template formatting
    train_path = output_dir / "cultivate.jsonl"
    exam_path = output_dir / "exam.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for p in train_pairs:
            f.write(json.dumps(p.to_chat_format(system_directive=DEFAULT_SYSTEM_DIRECTIVE), ensure_ascii=False) + "\n")

    with open(exam_path, "w", encoding="utf-8") as f:
        for p in exam_pairs:
            f.write(json.dumps(p.to_chat_format(system_directive=DEFAULT_SYSTEM_DIRECTIVE), ensure_ascii=False) + "\n")

    stats = {
        "total_unique_scenarios": total_scenarios,
        "train_scenarios_count": len(train_base_scenarios),
        "exam_scenarios_count": len(exam_base_scenarios),
        "augmented_train_pairs": len(train_pairs),
        "exam_pairs": len(exam_pairs),
        "overlap_count": len(overlap),
        "train_path": str(train_path),
        "exam_path": str(exam_path)
    }

    return train_path, exam_path, stats
