"""
Bonsai CLI - The art of cultivating massive foundation models into pocket-sized intelligences on consumer silicon.
"""

import os
import sys

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from datetime import datetime
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from bonsai.core.constants import BONSAI_VERSION, get_resolved_model_path, MANIFEST_FILENAME
from bonsai.core.probe import probe_hardware
from bonsai.core.ingest import ingest_source
from bonsai.core.teacher import TeacherEngine
from bonsai.core.holdout import cultivate_and_split_zero_leakage
from bonsai.core.manifest import BonsaiManifest, load_manifest
from bonsai.trainers.trainer import BonsaiTrainer
from bonsai.evaluators.exam import BonsaiExam
from bonsai.evaluators.reporter import render_terminal_scorecard, export_svg_report, export_html_report
from bonsai.exporters.ollama import export_ollama_modelfile, generate_local_runner

app = typer.Typer(
    name="bonsai",
    help="Cultivate massive foundation models into pocket-sized intelligences on consumer silicon.",
    add_completion=False
)

console = Console(force_terminal=True, legacy_windows=False)

BANNER = r"""[bold green]
   ___  ____  _  _ ____   __   ____ 
  / __)(  _ \( \( ) ___)/ _\ (_  _)
 ( (__  ) _ ( )  ( \___ \/    \  )(  
  \___)(____/(_)\_)(____/\_/\_/ (__) 
[/bold green]
[dim green]The art of cultivating massive foundation models into pocket-sized intelligences.[/dim green]
"""

@app.command("probe")
def cmd_probe():
    """Inspects host hardware telemetry, detects available VRAM, and recommends the zero-OOM micro-model."""
    console.print(BANNER)
    console.print("[bold cyan]🔍 Probing consumer silicon telemetry...[/bold cyan]\n")

    profile = probe_hardware()

    table = Table(title="💻 Machine Telemetry & Auto-Downshift Profile", border_style="green", header_style="bold green")
    table.add_column("Property", style="bold white", width=28)
    table.add_column("Detected / Configured", style="cyan", width=45)

    table.add_row("Operating System", f"{profile.os_name} {profile.os_version}")
    table.add_row("Processor (CPU)", f"{profile.cpu_model} ({profile.cpu_cores_physical}P / {profile.cpu_cores_logical}L cores)")
    table.add_row("Host System RAM", f"{profile.system_ram_gb} GB")
    table.add_row("GPU Hardware", f"{profile.gpu_name} ({profile.gpu_type.upper()})")
    table.add_row("Video Memory (VRAM)", f"{profile.vram_free_mb:.0f} MB Free / {profile.vram_total_mb:.0f} MB Total")
    table.add_row("Recommended Micro-Model", f"[bold yellow]{profile.recommended_model}[/bold yellow]")
    table.add_row("Zero-OOM Batching", f"Batch {profile.recommended_batch_size} × Accum {profile.recommended_grad_accum} (Virtual {profile.recommended_batch_size * profile.recommended_grad_accum})")
    table.add_row("Synaptic Rank / Alpha", f"LoRA r={profile.recommended_lora_rank}, α={profile.recommended_lora_alpha}")
    table.add_row("Focus Aperture", f"{profile.recommended_max_seq_len} Tokens Max Context")
    table.add_row("Precision Backend", f"{profile.precision.upper()}")

    console.print(table)

    console.print("\n[bold green]🌱 Bonsai Human Concept Translation Layer:[/bold green]")
    for concept, meaning in profile.jargon_translations.items():
        console.print(f"  • [bold white]{concept}:[/bold white] [green]{meaning}[/green]")
    console.print("")

@app.command("prune")
def cmd_prune(
    source: str = typer.Argument(..., help="Path to raw folder, file, logs, or plain-English task specification"),
    output_dir: str = typer.Option("./bonsai_pruned", "--output-dir", "-o", help="Target output directory"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Base micro-model to prune (defaults to auto-probed model)"),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Study passes over dataset"),
    holdout_ratio: float = typer.Option(0.15, "--holdout", "-h", help="Ratio for blind holdout exam (default 15%)")
):
    """Cultivates raw data into a pocket-sized micro-AI, fine-tunes locally, and runs the blind proof exam."""
    console.print(BANNER)
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Hardware Probe
        console.print("[bold green]Step 1/5 • Hardware Auto-Probe[/bold green]")
        profile = probe_hardware()
        target_model = get_resolved_model_path(model) if model else profile.recommended_model
        console.print(f"  ✓ Target Silicon: [cyan]{profile.gpu_name}[/cyan] ({profile.vram_free_mb:.0f} MB VRAM)")
        console.print(f"  ✓ Selected Micro-Model: [bold yellow]{target_model}[/bold yellow] (Zero-OOM Protected)\n")

        # 2. Ingestion
        console.print("[bold green]Step 2/5 • Universal Ingestion[/bold green]")
        blocks = ingest_source(source)
        console.print(f"  ✓ Ingested [cyan]{len(blocks)}[/cyan] raw document chunks from [white]{source}[/white]\n")

        # 3. Teacher Cultivation & Zero-Leakage Holdout Separation
        console.print("[bold green]Step 3/5 • Teacher Cultivation & Zero-Leakage Split[/bold green]")
        teacher = TeacherEngine()
        console.print(f"  ✓ Active Teacher Engine: [cyan]{teacher.active_backend}[/cyan]")

        train_path, exam_path, stats = cultivate_and_split_zero_leakage(
            blocks=blocks,
            output_dir=out_path,
            teacher=teacher,
            holdout_ratio=holdout_ratio,
            target_train_size=24,
            seed=42
        )
        console.print(f"  ✓ Unique Scenarios: [cyan]{stats['total_unique_scenarios']}[/cyan] (Verified Scenario Overlap: [bold green]0[/bold green])")
        console.print(f"  ✓ Training Set: [green]{stats['augmented_train_pairs']} samples[/green] (from {stats['train_scenarios_count']} scenarios) → [dim]{train_path.name}[/dim]")
        console.print(f"  ✓ Blind Holdout Exam: [magenta]{stats['exam_pairs']} samples[/magenta] (pristine unseen) → [dim]{exam_path.name}[/dim]\n")

        # 4. Pocket LoRA Distillation
        console.print("[bold green]Step 4/5 • Pocket Distillation (Local Training)[/bold green]")
        trainer = BonsaiTrainer(
            model_name=target_model,
            output_dir=str(out_path),
            epochs=epochs,
            batch_size=profile.recommended_batch_size,
            grad_accum_steps=profile.recommended_grad_accum,
            lora_rank=profile.recommended_lora_rank,
            lora_alpha=profile.recommended_lora_alpha,
            max_seq_len=profile.recommended_max_seq_len,
            device_type=profile.gpu_type
        )
        train_telemetry = trainer.train(str(train_path))

        # Export Modelfile & Local Runner
        modelfile_path = out_path / "Modelfile"
        export_ollama_modelfile(target_model, str(out_path / "adapter"), str(modelfile_path))
        runner_path = out_path / "serve.py"
        generate_local_runner(target_model, str(out_path / "adapter"), str(runner_path))
        console.print(f"  ✓ Ollama Modelfile generated: [cyan]{modelfile_path}[/cyan]")
        console.print(f"  ✓ Instant Local Runner generated: [cyan]{runner_path}[/cyan]\n")

        # 5. Automated Proof Exam
        console.print("[bold green]Step 5/5 • Blind Holdout Proof Machine[/bold green]")
        exam = BonsaiExam(
            base_model_name=target_model,
            adapter_path=str(out_path / "adapter"),
            device=profile.gpu_type
        )
        scorecard = exam.run_exam(str(exam_path))

        # Render Reports
        svg_path = out_path / "bonsai-report.svg"
        html_path = out_path / "bonsai-report.html"
        render_terminal_scorecard(scorecard)
        export_svg_report(scorecard, str(svg_path))
        export_html_report(scorecard, str(html_path))

        # Save Canonical Manifest
        manifest = BonsaiManifest(
            version=BONSAI_VERSION,
            created_at=datetime.now().isoformat(),
            output_dir=str(out_path),
            base_model=target_model,
            adapter_dir=str(out_path / "adapter"),
            train_dataset=str(train_path),
            exam_dataset=str(exam_path),
            modelfile=str(modelfile_path),
            runner_script=str(runner_path),
            report_svg=str(svg_path),
            report_html=str(html_path),
            hardware={
                "gpu_name": profile.gpu_name,
                "gpu_type": profile.gpu_type,
                "vram_total_mb": profile.vram_total_mb,
                "vram_free_mb": profile.vram_free_mb
            },
            training_telemetry=train_telemetry,
            exam_scorecard={
                "base_schema_validity_pct": scorecard.base_schema_validity_pct,
                "bonsai_schema_validity_pct": scorecard.bonsai_schema_validity_pct,
                "base_task_accuracy_pct": scorecard.base_task_accuracy_pct,
                "bonsai_task_accuracy_pct": scorecard.bonsai_task_accuracy_pct,
                "base_hallucination_pct": scorecard.base_hallucination_pct,
                "bonsai_hallucination_pct": scorecard.bonsai_hallucination_pct,
                "base_tokens_sec": scorecard.base_tokens_sec,
                "bonsai_tokens_sec": scorecard.bonsai_tokens_sec,
                "peak_vram_mb": scorecard.peak_vram_mb
            }
        )
        manifest_file = manifest.save(out_path)
        console.print(f"[bold green]📋 State Manifest saved:[/bold green] [cyan]{manifest_file}[/cyan]\n")

    except Exception as e:
        console.print(f"\n[bold red]❌ Bonsai Error:[/bold red] {e}\n")
        raise typer.Exit(code=1)

@app.command("exam")
def cmd_exam(
    target: str = typer.Argument("./bonsai_pruned", help="Path to output directory or bonsai_manifest.json"),
    exam_path: Optional[str] = typer.Option(None, "--exam-path", "-e", help="Optional override path to exam.jsonl"),
    base_model: Optional[str] = typer.Option(None, "--base-model", "-m", help="Optional override for base model"),
    adapter: Optional[str] = typer.Option(None, "--adapter", "-a", help="Optional override for adapter path")
):
    """Runs the Proof Machine against unseen blind exam.jsonl using the run manifest."""
    console.print(BANNER)
    try:
        manifest = load_manifest(target)
        target_exam = exam_path or manifest.exam_dataset
        target_base = base_model or manifest.base_model
        target_adapter = adapter or manifest.adapter_dir

        profile = probe_hardware()
        exam = BonsaiExam(
            base_model_name=target_base,
            adapter_path=target_adapter,
            device=profile.gpu_type
        )
        scorecard = exam.run_exam(target_exam)

        out_p = Path(manifest.output_dir)
        render_terminal_scorecard(scorecard)
        export_svg_report(scorecard, str(out_p / "bonsai-report.svg"))
        export_html_report(scorecard, str(out_p / "bonsai-report.html"))

    except Exception as e:
        console.print(f"\n[bold red]❌ Bonsai Exam Error:[/bold red] {e}\n")
        raise typer.Exit(code=1)

@app.command("serve")
def cmd_serve(
    target: str = typer.Argument("./bonsai_pruned", help="Path to output directory or bonsai_manifest.json")
):
    """Starts an interactive local terminal session using the run manifest."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from bonsai.core.constants import DEFAULT_SYSTEM_DIRECTIVE

    console.print(BANNER)
    try:
        manifest = load_manifest(target)
        base_model = manifest.base_model
        adapter_path = manifest.adapter_dir

        console.print(f"[bold green]🌱 Loading Bonsai Micro-AI from {adapter_path}...[/bold green]")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, trust_remote_code=True).to(device)
        model = PeftModel.from_pretrained(base, adapter_path).to(device)
        model.eval()

        console.print("[bold cyan]✨ Bonsai ready for live queries! Type 'exit' to quit:[/bold cyan]\n")
        while True:
            try:
                query = console.input("[bold green]Query > [/bold green]")
                if query.strip().lower() in ("exit", "quit"):
                    break
                messages = [
                    {"role": "system", "content": DEFAULT_SYSTEM_DIRECTIVE},
                    {"role": "user", "content": query}
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                gen = out[0][inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(gen, skip_special_tokens=True).strip()
                console.print(Panel(response, title="[bold green]Bonsai Micro-AI[/bold green]", border_style="green"))
            except (KeyboardInterrupt, EOFError):
                break

    except Exception as e:
        console.print(f"\n[bold red]❌ Bonsai Serve Error:[/bold red] {e}\n")
        raise typer.Exit(code=1)

@app.command("ui")
def cmd_ui(
    port: int = typer.Option(7860, "--port", "-p", help="Port to host web UI on")
):
    """Launches the standalone dark-mode Web UI chat and opens it in your default browser."""
    from bonsai.web_server import run_server
    run_server()

if __name__ == "__main__":
    app()
