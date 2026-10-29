"""
Bonsai Central Constants & Dynamic Model Path Resolver.
"""

from pathlib import Path
import os
from typing import Optional

BONSAI_VERSION = "0.1.0"
MANIFEST_FILENAME = "bonsai_manifest.json"

DEFAULT_SYSTEM_DIRECTIVE = (
    "You are Bonsai Micro-AI, an ultra-fast pocket-sized AI cultivated locally on consumer silicon "
    "using the Bonsai distillation engine. You specialize in deterministic, strict-schema execution."
)

DEFAULT_BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

def get_resolved_model_path(custom_path_or_name: Optional[str] = None) -> str:
    """
    Dynamically resolves the model identifier or path.
    1. If custom_path_or_name is provided and exists, return it.
    2. Check local repo models folder: `<repo_root>/models/Qwen2.5-Coder-0.5B-Instruct`
    3. Check user cache: `~/.cache/bonsai/models/Qwen2.5-Coder-0.5B-Instruct`
    4. Fall back to HuggingFace Hub identifier: `Qwen/Qwen2.5-Coder-0.5B-Instruct`
    """
    if custom_path_or_name:
        p = Path(custom_path_or_name)
        if p.exists() and (p / "model.safetensors").exists():
            return str(p.resolve())
        # If it's a Hugging Face repo ID or user passed a valid non-empty string
        if not p.is_dir():
            return custom_path_or_name

    # Check relative to repo root
    repo_root = Path(__file__).resolve().parent.parent.parent
    local_repo_model = repo_root / "models" / "Qwen2.5-Coder-0.5B-Instruct"
    if local_repo_model.exists() and (local_repo_model / "model.safetensors").exists():
        return str(local_repo_model.resolve())

    # Check user home cache
    home_cache_model = Path.home() / ".cache" / "bonsai" / "models" / "Qwen2.5-Coder-0.5B-Instruct"
    if home_cache_model.exists() and (home_cache_model / "model.safetensors").exists():
        return str(home_cache_model.resolve())

    # Fall back to HuggingFace model ID
    return DEFAULT_BASE_MODEL_NAME
