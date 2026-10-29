"""
Pocket Distillation & LoRA Training Engine for Bonsai.
Executes real local fine-tuning on consumer silicon with Zero-OOM protection and live telemetry.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from rich.console import Console
from bonsai.core.constants import DEFAULT_SYSTEM_DIRECTIVE

console = Console(force_terminal=True, legacy_windows=False)

class BonsaiTrainer:
    def __init__(
        self,
        model_name: str,
        output_dir: str = "./bonsai_pruned",
        epochs: int = 3,
        batch_size: int = 1,
        grad_accum_steps: int = 4,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        learning_rate: float = 2e-4,
        max_seq_len: int = 512,
        device_type: str = "cuda"
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.max_seq_len = max_seq_len
        self.device_type = device_type

    def train(self, dataset_path: str, progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        Executes real local LoRA training pass with genuine gradient computation and prompt token loss masking.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType

        start_time = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = self.output_dir / "adapter"

        # Determine target device
        device = "cuda" if (self.device_type == "cuda" and torch.cuda.is_available()) else ("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32

        console.print(f"[bold green]🌱 [Bonsai Distiller][/bold green] Initializing base micro-model: [cyan]{self.model_name}[/cyan]")
        console.print(f"[dim]Device: {device.upper()} | Precision: {dtype} | Rank: {self.lora_rank} | Alpha: {self.lora_alpha}[/dim]")

        # 1. Load Tokenizer & Model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # Enable gradient checkpointing for rock-solid memory saving
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        # 2. Configure PEFT LoRA
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none"
        )
        peft_model = get_peft_model(model, lora_config)
        peft_model.to(device)

        trainable_params, all_param = peft_model.get_nb_trainable_parameters()
        pct = (trainable_params / all_param) * 100
        console.print(f"[bold cyan]🎯 Synaptic Plasticity:[/bold cyan] {trainable_params:,} / {all_param:,} parameters ({pct:.2f}% active branches)")

        # 3. Load & Tokenize Dataset with Prompt Token Loss Masking
        tokenized_samples = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                messages = item.get("messages", [])

                # Full conversation formatted
                full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                full_enc = tokenizer(
                    full_text,
                    truncation=True,
                    max_length=self.max_seq_len,
                    padding="max_length",
                    return_tensors="pt"
                )
                input_ids = full_enc["input_ids"].squeeze(0)
                attention_mask = full_enc["attention_mask"].squeeze(0)

                # Prompt-only text to calculate prompt token length
                prompt_messages = [m for m in messages if m["role"] != "assistant"]
                prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                prompt_enc = tokenizer(prompt_text, truncation=True, max_length=self.max_seq_len, add_special_tokens=False)
                prompt_len = min(len(prompt_enc["input_ids"]), self.max_seq_len)

                # Labels: Mask user prompt tokens and padding with -100
                labels = input_ids.clone()
                labels[:prompt_len] = -100
                labels[attention_mask == 0] = -100

                tokenized_samples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                })

        if not tokenized_samples:
            raise ValueError(f"No valid training samples found in {dataset_path}")

        # 4. Training Loop with Live Telemetry
        optimizer = torch.optim.AdamW(peft_model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        total_steps = max(1, (len(tokenized_samples) * self.epochs) // self.grad_accum_steps)
        step_count = 0
        loss_history = []
        initial_loss = None
        final_loss = None

        peft_model.train()
        optimizer.zero_grad()

        console.print(f"[bold green]🚀 Commencing Pocket Distillation:[/bold green] {len(tokenized_samples)} samples × {self.epochs} passes ({total_steps} optimizer steps)")

        accum_loss = 0.0
        accum_count = 0

        for epoch in range(self.epochs):
            for idx, sample in enumerate(tokenized_samples):
                input_ids = sample["input_ids"].unsqueeze(0).to(device)
                attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
                labels = sample["labels"].unsqueeze(0).to(device)

                outputs = peft_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / self.grad_accum_steps
                loss.backward()

                current_loss_val = outputs.loss.item()
                accum_loss += current_loss_val
                accum_count += 1

                if initial_loss is None:
                    initial_loss = current_loss_val

                if (idx + 1) % self.grad_accum_steps == 0 or (idx + 1) == len(tokenized_samples):
                    torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    step_count += 1

                    avg_loss = accum_loss / max(accum_count, 1)
                    loss_history.append(avg_loss)
                    final_loss = avg_loss
                    accum_loss = 0.0
                    accum_count = 0

                    vram_str = "0 MB"
                    if device == "cuda":
                        allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
                        vram_str = f"{allocated:.0f} MB"

                    pct = (step_count / total_steps) * 100
                    console.print(f"  [cyan]Step {step_count:2d}/{total_steps}[/cyan] ({pct:>3.0f}%) | Epoch {epoch+1}/{self.epochs} | Loss: [yellow]{avg_loss:.4f}[/yellow] | VRAM: [magenta]{vram_str}[/magenta]")

                    if progress_callback:
                        progress_callback({
                            "step": step_count,
                            "total_steps": total_steps,
                            "loss": avg_loss,
                            "vram": vram_str
                        })

        # 5. Save Adapter & Artifacts
        peft_model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(self.output_dir))

        peak_vram = 0.0
        if device == "cuda":
            peak_vram = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)

        duration = round(time.time() - start_time, 2)
        loss_reduction = round(((initial_loss - final_loss) / initial_loss) * 100, 1) if initial_loss and initial_loss > 0 else 0.0

        telemetry = {
            "model_name": self.model_name,
            "adapter_path": str(adapter_dir),
            "output_dir": str(self.output_dir),
            "epochs": self.epochs,
            "total_steps": step_count,
            "initial_loss": round(initial_loss or 0.0, 4),
            "final_loss": round(final_loss or 0.0, 4),
            "loss_reduction_pct": loss_reduction,
            "duration_seconds": duration,
            "peak_vram_mb": peak_vram,
            "device": device
        }

        # Clear GPU cache after training pass
        if device == "cuda":
            torch.cuda.empty_cache()

        console.print(f"[bold green]✨ Cultivation complete![/bold green] Adapter saved to [cyan]{adapter_dir}[/cyan]")
        console.print(f"[bold green]📉 Loss Drop:[/bold green] {telemetry['initial_loss']} → {telemetry['final_loss']} ([bold yellow]-{loss_reduction}%[/bold yellow]) in {duration}s")

        return telemetry
