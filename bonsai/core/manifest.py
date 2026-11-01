"""
Unified Manifest Architecture for Bonsai.
Provides single-source-of-truth state handoff across probe, prune, exam, merge, and serve.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from bonsai.core.constants import BONSAI_VERSION, MANIFEST_FILENAME

@dataclass
class BonsaiManifest:
    version: str
    created_at: str
    output_dir: str
    base_model: str
    adapter_dir: str
    train_dataset: str
    exam_dataset: str
    modelfile: str
    runner_script: str
    report_svg: str
    report_html: str
    hardware: Dict[str, Any]
    training_telemetry: Dict[str, Any]
    exam_scorecard: Optional[Dict[str, Any]] = None
    merged_model_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, output_dir: Optional[Union[str, Path]] = None) -> Path:
        target_dir = Path(output_dir) if output_dir else Path(self.output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = target_dir / MANIFEST_FILENAME
        manifest_file.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return manifest_file

def load_manifest(path_or_dir: Union[str, Path]) -> BonsaiManifest:
    """Loads a BonsaiManifest from a file path or an output directory."""
    p = Path(path_or_dir)
    if p.is_dir():
        p = p / MANIFEST_FILENAME

    if not p.exists():
        raise FileNotFoundError(f"Bonsai manifest not found at: {p}. Run 'bonsai prune' first to generate a run manifest.")

    data = json.loads(p.read_text(encoding="utf-8"))
    
    # Filter known fields for backward and forward compatibility
    known_fields = {
        "version", "created_at", "output_dir", "base_model", "adapter_dir",
        "train_dataset", "exam_dataset", "modelfile", "runner_script",
        "report_svg", "report_html", "hardware", "training_telemetry",
        "exam_scorecard", "merged_model_dir"
    }
    filtered_data = {k: v for k, v in data.items() if k in known_fields}
    return BonsaiManifest(**filtered_data)
