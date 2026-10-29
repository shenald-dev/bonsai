"""
Tests for Bonsai Hardware Probe & Auto-Downshift Telemetry.
"""

from bonsai.core.probe import probe_hardware, HardwareProfile

def test_probe_hardware_structure():
    profile = probe_hardware()
    assert isinstance(profile, HardwareProfile)
    assert profile.os_name in ("Windows", "Linux", "Darwin")
    assert profile.cpu_cores_physical > 0
    assert profile.cpu_cores_logical >= profile.cpu_cores_physical
    assert profile.system_ram_gb > 0
    assert profile.gpu_type in ("cuda", "mps", "cpu")
    assert profile.recommended_model != ""
    assert profile.recommended_batch_size >= 1
    assert profile.recommended_grad_accum >= 1
    assert profile.recommended_lora_rank in (4, 8, 16)
    assert profile.recommended_lora_alpha in (8, 16, 32)
    assert profile.recommended_max_seq_len >= 384
    assert profile.precision in ("float16", "bfloat16", "float32")

def test_probe_jargon_translations():
    profile = probe_hardware()
    assert len(profile.jargon_translations) >= 5
    assert "Learning Agility (Learning Rate)" in profile.jargon_translations
    assert "Synaptic Plasticity (LoRA Rank & Alpha)" in profile.jargon_translations
    assert "Study Passes (Epochs)" in profile.jargon_translations
    assert "Zero-OOM Safety Guard" in profile.jargon_translations
