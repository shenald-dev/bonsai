"""
Hardware Telemetry & Zero-OOM Auto-Downshift Probe for Bonsai.
"""

from dataclasses import dataclass
import os
import platform
import shutil
import subprocess
from typing import Dict, Any, Optional
import psutil
from bonsai.core.constants import get_resolved_model_path

@dataclass
class HardwareProfile:
    os_name: str
    os_version: str
    cpu_model: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    system_ram_gb: float
    gpu_type: str  # "cuda", "mps", "cpu"
    gpu_name: str
    vram_total_mb: float
    vram_free_mb: float
    recommended_model: str
    recommended_batch_size: int
    recommended_grad_accum: int
    recommended_lora_rank: int
    recommended_lora_alpha: int
    recommended_max_seq_len: int
    precision: str
    jargon_translations: Dict[str, str]

def probe_hardware() -> HardwareProfile:
    """Probes the host machine's hardware capabilities and selects the optimal micro-model."""
    os_name = platform.system()
    os_version = platform.release()
    cpu_cores_phy = psutil.cpu_count(logical=False) or 4
    cpu_cores_log = psutil.cpu_count(logical=True) or 8
    cpu_model = platform.processor() or "Modern x86_64 / ARM CPU"
    total_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)

    gpu_type = "cpu"
    gpu_name = "CPU Backend (Fallback)"
    vram_total_mb = 0.0
    vram_free_mb = 0.0

    # 1. Probe CUDA via torch if available
    torch_detected = False
    try:
        import torch
        if torch.cuda.is_available():
            gpu_type = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            vram_total_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 0)
            mem_free, mem_total = torch.cuda.mem_get_info()
            vram_free_mb = round(mem_free / (1024 ** 2), 0)
            torch_detected = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_type = "mps"
            gpu_name = "Apple Silicon Unified Memory"
            vram_total_mb = total_ram_gb * 1024
            vram_free_mb = round(psutil.virtual_memory().available / (1024 ** 2), 0)
            torch_detected = True
    except Exception:
        pass

    # 2. Subprocess probe for NVIDIA if torch wasn't loaded or torch has no cuda yet
    if not torch_detected and shutil.which("nvidia-smi"):
        try:
            smi_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                encoding="utf-8",
                stderr=subprocess.DEVNULL
            ).strip()
            if smi_out:
                lines = smi_out.splitlines()
                first = lines[0].split(",")
                gpu_name = first[0].strip()
                vram_total_mb = float(first[1].strip())
                vram_free_mb = float(first[2].strip())
                gpu_type = "cuda"
        except Exception:
            pass

    # Model Selection & Zero-OOM Downshift Logic
    resolved_05b_model = get_resolved_model_path()

    if gpu_type == "cuda":
        # Consumer GPU logic
        if vram_free_mb < 3800:
            # 4GB card or congested memory (like RTX 2050 / GTX 1650) -> Auto-downshift to 0.5B
            recommended_model = resolved_05b_model
            recommended_batch_size = 1
            recommended_grad_accum = 4
            recommended_lora_rank = 8
            recommended_lora_alpha = 16
            recommended_max_seq_len = 512
            precision = "float16"
        elif vram_free_mb < 7500:
            # 6GB - 8GB card (e.g. RTX 3060 / 4060) -> 1.5B
            recommended_model = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
            recommended_batch_size = 2
            recommended_grad_accum = 4
            recommended_lora_rank = 16
            recommended_lora_alpha = 32
            recommended_max_seq_len = 1024
            precision = "float16"
        else:
            # 8GB - 12GB+ card -> 3B
            recommended_model = "Qwen/Qwen2.5-3B-Instruct"
            recommended_batch_size = 4
            recommended_grad_accum = 2
            recommended_lora_rank = 16
            recommended_lora_alpha = 32
            recommended_max_seq_len = 2048
            precision = "bfloat16"
    elif gpu_type == "mps":
        # Apple Silicon unified memory
        if total_ram_gb <= 16:
            recommended_model = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
            recommended_batch_size = 2
            recommended_grad_accum = 4
            recommended_lora_rank = 16
            recommended_lora_alpha = 32
            recommended_max_seq_len = 1024
            precision = "float16"
        else:
            recommended_model = "Qwen/Qwen2.5-3B-Instruct"
            recommended_batch_size = 4
            recommended_grad_accum = 2
            recommended_lora_rank = 16
            recommended_lora_alpha = 32
            recommended_max_seq_len = 2048
            precision = "bfloat16"
    else:
        # CPU Fallback
        recommended_model = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
        recommended_batch_size = 1
        recommended_grad_accum = 2
        recommended_lora_rank = 4
        recommended_lora_alpha = 8
        recommended_max_seq_len = 384
        precision = "float32"

    jargon_translations = {
        "Learning Agility (Learning Rate)": "2e-4 (Adaptive Micro-Pacing)",
        "Synaptic Plasticity (LoRA Rank & Alpha)": f"Rank {recommended_lora_rank} / Alpha {recommended_lora_alpha} (Targeted Synapse Pruning)",
        "Study Passes (Epochs)": "3 Passes (Rapid Skill Consolidation)",
        "Focus Aperture (Max Token Context)": f"{recommended_max_seq_len} Tokens",
        "Attention Batching (Grad Accumulation)": f"Virtual Step of {recommended_batch_size * recommended_grad_accum}",
        "Zero-OOM Safety Guard": f"Active ({vram_free_mb:.0f} MB Available / Headroom Protected)"
    }

    return HardwareProfile(
        os_name=os_name,
        os_version=os_version,
        cpu_model=cpu_model,
        cpu_cores_physical=cpu_cores_phy,
        cpu_cores_logical=cpu_cores_log,
        system_ram_gb=total_ram_gb,
        gpu_type=gpu_type,
        gpu_name=gpu_name,
        vram_total_mb=vram_total_mb,
        vram_free_mb=vram_free_mb,
        recommended_model=recommended_model,
        recommended_batch_size=recommended_batch_size,
        recommended_grad_accum=recommended_grad_accum,
        recommended_lora_rank=recommended_lora_rank,
        recommended_lora_alpha=recommended_lora_alpha,
        recommended_max_seq_len=recommended_max_seq_len,
        precision=precision,
        jargon_translations=jargon_translations
    )
