"""
Bonsai LoRA Adapter Fusion & Merging Engine.
Fuses trained PEFT synaptic adapter weights back into base foundation model weights,
producing a standalone, unquantized or quantized, zero-dependency model.
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console

console = Console(force_terminal=True, legacy_windows=False)


class BonsaiMerger:
    """Fuses LoRA adapter weights directly into the base model weights."""

    def __init__(
        self,
        base_model: str,
        adapter_path: str,
        output_dir: str = "./bonsai_pruned/merged",
        device: str = "auto"
    ):
        self.base_model = base_model
        self.adapter_path = Path(adapter_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.device = device

    def merge(self) -> Dict[str, Any]:
        """
        Executes genuine weight merging via PEFT merge_and_unload().
        Saves standalone safetensors and tokenizer.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        start_time = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Select target device
        if self.device == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            target_device = self.device

        dtype = torch.float16 if target_device == "cuda" else torch.float32

        console.print(f"[bold green]🌱 [Bonsai Merger][/bold green] Loading base model: [cyan]{self.base_model}[/cyan]")
        console.print(f"[dim]Device: {target_device.upper()} | Precision: {dtype}[/dim]")

        # 1. Load Tokenizer & Base Model
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(target_device)

        # 2. Attach Adapter
        console.print(f"[bold green]🌿 [Bonsai Merger][/bold green] Attaching adapter weights from: [cyan]{self.adapter_path}[/cyan]")
        peft_model = PeftModel.from_pretrained(base, str(self.adapter_path)).to(target_device)

        # 3. Fuse & Unload
        console.print("[bold cyan]⚡ Fusing synaptic weights into base matrix (merge_and_unload)...[/bold cyan]")
        merged_model = peft_model.merge_and_unload()

        # 4. Save Standalone Merged Model
        console.print(f"[bold green]💾 Exporting standalone model to:[/bold green] [cyan]{self.output_dir}[/cyan]")
        merged_model.save_pretrained(
            str(self.output_dir),
            safe_serialization=True
        )
        tokenizer.save_pretrained(str(self.output_dir))

        # Copy over generation config or chat template if present in base or adapter
        for candidate in [self.adapter_path, self.adapter_path.parent]:
            jinja_file = candidate / "chat_template.jinja"
            if jinja_file.exists():
                shutil.copy2(jinja_file, self.output_dir / "chat_template.jinja")
                break

        duration = round(time.time() - start_time, 2)
        total_size_bytes = sum(f.stat().st_size for f in self.output_dir.glob("*") if f.is_file())
        total_size_mb = round(total_size_bytes / (1024 ** 2), 1)

        telemetry = {
            "base_model": self.base_model,
            "adapter_path": str(self.adapter_path),
            "merged_output_dir": str(self.output_dir),
            "duration_seconds": duration,
            "total_size_mb": total_size_mb,
            "device": target_device
        }

        console.print(f"[bold green]✨ Fusion complete in {duration}s! Total size: {total_size_mb} MB[/bold green]")
        console.print("[dim]Model is now 100% standalone and ready for llama.cpp/GGUF quantization or zero-overhead inference.[/dim]\n")

        return telemetry
