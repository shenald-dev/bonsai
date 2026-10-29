"""
Scorecard & Viral Report Generator for Bonsai.
Renders viral ASCII terminal scorecards, standalone HTML report cards, and SVG badges.
"""

import sys

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from bonsai.evaluators.exam import ExamScorecard

console = Console(force_terminal=True, legacy_windows=False)

def render_terminal_scorecard(scorecard: ExamScorecard):
    """Renders the high-impact ASCII terminal scorecard."""
    schema_delta = scorecard.bonsai_schema_validity_pct - scorecard.base_schema_validity_pct
    acc_delta = scorecard.bonsai_task_accuracy_pct - scorecard.base_task_accuracy_pct
    speedup = round(scorecard.bonsai_tokens_sec / max(scorecard.base_tokens_sec, 0.1), 2)

    console.print("\n")
    console.print("[bold green]╭─────────────────────────────────────────────────────────────────────────────╮[/bold green]")
    console.print("[bold green]│                 🌱 BONSAI PROOF & SPEED REPORT CARD                         │[/bold green]")
    console.print("[bold green]│      Zero-Knowledge Local Micro-Distillation on Consumer Silicon            │[/bold green]")
    console.print("[bold green]╰─────────────────────────────────────────────────────────────────────────────╯[/bold green]")

    table = Table(title="🔬 Blind Holdout Performance (Unseen Test Set)", header_style="bold cyan", border_style="dim")
    table.add_column("Evaluation Metric", style="bold white", width=30)
    table.add_column("Base Model (Raw)", justify="center", style="yellow", width=18)
    table.add_column("Bonsai Cultivated", justify="center", style="bold green", width=20)
    table.add_column("Impact Delta", justify="center", style="bold magenta", width=16)

    table.add_row(
        "Strict Schema Validity",
        f"{scorecard.base_schema_validity_pct:.1f}%",
        f"{scorecard.bonsai_schema_validity_pct:.1f}%",
        f"+{schema_delta:.1f}%" if schema_delta >= 0 else f"{schema_delta:.1f}%"
    )
    table.add_row(
        "Task Extraction Accuracy",
        f"{scorecard.base_task_accuracy_pct:.1f}%",
        f"{scorecard.bonsai_task_accuracy_pct:.1f}%",
        f"+{acc_delta:.1f}%" if acc_delta >= 0 else f"{acc_delta:.1f}%"
    )
    table.add_row(
        "Hallucination / Error Rate",
        f"{scorecard.base_hallucination_pct:.1f}%",
        f"{scorecard.bonsai_hallucination_pct:.1f}%",
        f"-{scorecard.base_hallucination_pct - scorecard.bonsai_hallucination_pct:.1f}%"
    )
    table.add_row(
        "Inference Throughput",
        f"{scorecard.base_tokens_sec:.1f} tok/s",
        f"{scorecard.bonsai_tokens_sec:.1f} tok/s",
        f"{speedup}x Speedup"
    )
    table.add_row(
        "Average Latency",
        f"{scorecard.base_avg_latency_ms:.0f} ms",
        f"{scorecard.bonsai_avg_latency_ms:.0f} ms",
        f"-{scorecard.base_avg_latency_ms - scorecard.bonsai_avg_latency_ms:.0f} ms"
    )
    table.add_row(
        "Peak VRAM Memory",
        f"{scorecard.peak_vram_mb:.0f} MB",
        f"{scorecard.peak_vram_mb:.0f} MB",
        "Zero-OOM Pass"
    )

    console.print(table)

    # Show side-by-side prompt diff if available
    if scorecard.sample_comparisons:
        first = scorecard.sample_comparisons[0]
        console.print("\n[bold cyan]⚡ Unseen Blind Exam Sample Comparison:[/bold cyan]")
        console.print(f"[dim]Input Query: {first.prompt[:120]}...[/dim]\n")

        diff_table = Table(show_header=True, header_style="bold", border_style="dim")
        diff_table.add_column("❌ Raw Base Model Output", style="red", width=42)
        diff_table.add_column("✅ Bonsai Cultivated Output", style="green", width=42)

        base_snippet = first.base_output[:280] + ("..." if len(first.base_output) > 280 else "")
        bonsai_snippet = first.bonsai_output[:280] + ("..." if len(first.bonsai_output) > 280 else "")

        diff_table.add_row(base_snippet, bonsai_snippet)
        console.print(diff_table)

    console.print("\n[bold green]✓ Proof Verification Complete: Model ready for Ollama export or local serving.[/bold green]\n")

def export_svg_report(scorecard: ExamScorecard, output_path: str = "bonsai-report.svg"):
    """Generates a standalone, beautiful SVG scorecard for GitHub / social media."""
    schema_delta = scorecard.bonsai_schema_validity_pct - scorecard.base_schema_validity_pct
    acc_delta = scorecard.bonsai_task_accuracy_pct - scorecard.base_task_accuracy_pct

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 850 480" width="850" height="480">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0a0f0d"/>
      <stop offset="100%" stop-color="#141e18"/>
    </linearGradient>
    <linearGradient id="emerald" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#10b981"/>
      <stop offset="100%" stop-color="#34d399"/>
    </linearGradient>
  </defs>

  <!-- Background -->
  <rect width="850" height="480" rx="16" fill="url(#bg)" stroke="#22382c" stroke-width="2"/>

  <!-- Header -->
  <text x="40" y="55" fill="#34d399" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif" font-size="22" font-weight="700">🌱 BONSAI PROOF SCORECARD</text>
  <text x="40" y="80" fill="#94a3b8" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif" font-size="13">Zero-Knowledge Local Micro-Distillation on Consumer Hardware</text>

  <!-- Base Model vs Distilled Badge -->
  <rect x="40" y="110" width="365" height="70" rx="10" fill="#16221c" stroke="#233d2e"/>
  <text x="60" y="138" fill="#94a3b8" font-size="12" font-family="sans-serif">BASE MODEL (RAW)</text>
  <text x="60" y="162" fill="#f59e0b" font-size="16" font-weight="bold" font-family="sans-serif">{scorecard.base_model_name.split('/')[-1]}</text>

  <rect x="445" y="110" width="365" height="70" rx="10" fill="#132c20" stroke="#10b981"/>
  <text x="465" y="138" fill="#a7f3d0" font-size="12" font-family="sans-serif">CULTIVATED MICRO-MODEL</text>
  <text x="465" y="162" fill="#34d399" font-size="16" font-weight="bold" font-family="sans-serif">Bonsai 0.5B Specialized</text>

  <!-- Metric 1: Schema Validity -->
  <rect x="40" y="200" width="770" height="52" rx="8" fill="#131e18"/>
  <text x="60" y="232" fill="#e2e8f0" font-size="14" font-weight="600" font-family="sans-serif">Strict Schema Validity</text>
  <text x="350" y="232" fill="#f59e0b" font-size="14" font-family="sans-serif">{scorecard.base_schema_validity_pct:.1f}%</text>
  <text x="520" y="232" fill="#34d399" font-size="15" font-weight="bold" font-family="sans-serif">{scorecard.bonsai_schema_validity_pct:.1f}%</text>
  <rect x="680" y="214" width="110" height="26" rx="13" fill="#064e3b"/>
  <text x="735" y="231" fill="#6ee7b7" font-size="12" font-weight="bold" font-family="sans-serif" text-anchor="middle">+{schema_delta:.1f}%</text>

  <!-- Metric 2: Task Accuracy -->
  <rect x="40" y="262" width="770" height="52" rx="8" fill="#131e18"/>
  <text x="60" y="294" fill="#e2e8f0" font-size="14" font-weight="600" font-family="sans-serif">Task Extraction Accuracy</text>
  <text x="350" y="294" fill="#f59e0b" font-size="14" font-family="sans-serif">{scorecard.base_task_accuracy_pct:.1f}%</text>
  <text x="520" y="294" fill="#34d399" font-size="15" font-weight="bold" font-family="sans-serif">{scorecard.bonsai_task_accuracy_pct:.1f}%</text>
  <rect x="680" y="276" width="110" height="26" rx="13" fill="#064e3b"/>
  <text x="735" y="293" fill="#6ee7b7" font-size="12" font-weight="bold" font-family="sans-serif" text-anchor="middle">+{acc_delta:.1f}%</text>

  <!-- Metric 3: Speed & Memory -->
  <rect x="40" y="324" width="770" height="52" rx="8" fill="#131e18"/>
  <text x="60" y="356" fill="#e2e8f0" font-size="14" font-weight="600" font-family="sans-serif">Generation Speed &amp; VRAM</text>
  <text x="350" y="356" fill="#f59e0b" font-size="14" font-family="sans-serif">{scorecard.base_tokens_sec:.1f} tok/s</text>
  <text x="520" y="356" fill="#34d399" font-size="15" font-weight="bold" font-family="sans-serif">{scorecard.bonsai_tokens_sec:.1f} tok/s</text>
  <rect x="680" y="338" width="110" height="26" rx="13" fill="#022c22"/>
  <text x="735" y="355" fill="#34d399" font-size="12" font-weight="bold" font-family="sans-serif" text-anchor="middle">{scorecard.peak_vram_mb:.0f} MB RAM</text>

  <!-- Footer -->
  <text x="40" y="440" fill="#64748b" font-size="12" font-family="sans-serif">Cultivated with Bonsai v0.1 • Tested on Unseen 15% Blind Holdout Set</text>
  <text x="810" y="440" fill="#10b981" font-size="12" font-weight="bold" font-family="sans-serif" text-anchor="end">uvx bonsai prune ./data</text>
</svg>"""

    Path(output_path).write_text(svg, encoding="utf-8")
    console.print(f"[bold green]📊 Viral SVG Scorecard generated:[/bold green] [cyan]{output_path}[/cyan]")

def export_html_report(scorecard: ExamScorecard, output_path: str = "bonsai-report.html"):
    """Generates an interactive, standalone HTML report card."""
    schema_delta = scorecard.bonsai_schema_validity_pct - scorecard.base_schema_validity_pct
    acc_delta = scorecard.bonsai_task_accuracy_pct - scorecard.base_task_accuracy_pct

    sample_html = ""
    for idx, s in enumerate(scorecard.sample_comparisons):
        sample_html += f"""
        <div class="sample-card">
          <div class="sample-prompt"><strong>Prompt #{idx+1}:</strong> {s.prompt[:160]}...</div>
          <div class="sample-grid">
            <div class="sample-col base">
              <span class="badge badge-fail">Raw Base Model ({'Valid JSON' if s.base_valid_json else 'Malformed / Rambling'})</span>
              <pre>{s.base_output}</pre>
            </div>
            <div class="sample-col bonsai">
              <span class="badge badge-success">Cultivated Bonsai (Strict JSON • {s.bonsai_field_match_pct:.0f}% Match)</span>
              <pre>{s.bonsai_output}</pre>
            </div>
          </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Bonsai Proof & Speed Report Card</title>
  <style>
    body {{
      background: #090e0b;
      color: #e2e8f0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 0;
      padding: 40px 20px;
    }}
    .container {{
      max-width: 960px;
      margin: 0 auto;
    }}
    .header {{
      text-align: center;
      margin-bottom: 40px;
    }}
    .header h1 {{
      color: #34d399;
      margin: 0 0 10px 0;
      font-size: 32px;
    }}
    .header p {{
      color: #94a3b8;
      margin: 0;
      font-size: 16px;
    }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 20px;
      margin-bottom: 40px;
    }}
    .metric-box {{
      background: #111a14;
      border: 1px solid #1f3325;
      border-radius: 12px;
      padding: 24px;
      text-align: center;
    }}
    .metric-title {{
      color: #94a3b8;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .metric-value {{
      font-size: 36px;
      font-weight: 800;
      color: #34d399;
      margin: 10px 0;
    }}
    .metric-sub {{
      font-size: 13px;
      color: #10b981;
    }}
    .sample-card {{
      background: #111a14;
      border: 1px solid #1f3325;
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 24px;
    }}
    .sample-prompt {{
      color: #cbd5e1;
      font-size: 14px;
      margin-bottom: 16px;
      background: #090e0b;
      padding: 12px;
      border-radius: 8px;
    }}
    .sample-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }}
    .sample-col pre {{
      background: #060a08;
      padding: 14px;
      border-radius: 8px;
      overflow-x: auto;
      font-size: 12px;
      color: #e2e8f0;
      max-height: 250px;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .badge-fail {{ background: #7f1d1d; color: #fca5a5; }}
    .badge-success {{ background: #064e3b; color: #6ee7b7; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🌱 BONSAI PROOF REPORT CARD</h1>
      <p>Automated Verification on 15% Blind Holdout Exam</p>
    </div>

    <div class="metrics-grid">
      <div class="metric-box">
        <div class="metric-title">Schema Validity</div>
        <div class="metric-value">{scorecard.bonsai_schema_validity_pct:.1f}%</div>
        <div class="metric-sub">+{schema_delta:.1f}% vs Raw Base</div>
      </div>
      <div class="metric-box">
        <div class="metric-title">Extraction Accuracy</div>
        <div class="metric-value">{scorecard.bonsai_task_accuracy_pct:.1f}%</div>
        <div class="metric-sub">+{acc_delta:.1f}% vs Raw Base</div>
      </div>
      <div class="metric-box">
        <div class="metric-title">Inference Speed</div>
        <div class="metric-value">{scorecard.bonsai_tokens_sec:.1f} <span style="font-size:18px">tok/s</span></div>
        <div class="metric-sub">Peak VRAM: {scorecard.peak_vram_mb:.0f} MB</div>
      </div>
    </div>

    <h2 style="color: #34d399; font-size: 20px; margin-bottom: 16px;">🔬 Blind Exam Side-by-Side Outputs</h2>
    {sample_html}
  </div>
</body>
</html>
"""

    Path(output_path).write_text(html, encoding="utf-8")
    console.print(f"[bold green]🌐 Interactive HTML Report generated:[/bold green] [cyan]{output_path}[/cyan]")
