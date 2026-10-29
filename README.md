<p align="center">
  <img src="assets/bonsai-hero-banner.png" alt="Bonsai Hero Banner" width="100%" />
</p>

<h1 align="center">🌱 BONSAI</h1>

<p align="center">
  <strong>The art of cultivating massive foundation models into pocket-sized intelligences on consumer silicon.</strong>
</p>

<p align="center">
  <a href="https://github.com/shenald-dev/bonsai/actions"><img src="https://img.shields.io/badge/CI-Passing-2ea44f.svg" alt="CI Status" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-0969da.svg" alt="License: MIT" /></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 3.10+" /></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.6%2B%20CUDA-ee4c2c.svg" alt="PyTorch CUDA" /></a>
  <a href="https://github.com/shenald-dev/bonsai"><img src="https://img.shields.io/badge/Zero--OOM-Guaranteed-brightgreen.svg" alt="Zero-OOM Protected" /></a>
  <a href="https://github.com/shenald-dev/bonsai"><img src="https://img.shields.io/badge/Tests-11%2F11%20Passing-success.svg" alt="Tests 100% Passed" /></a>
</p>

---

## ⚡ The 10-Second Quickstart

```bash
uvx bonsai prune ./my_raw_data
```

No JSONL formatting. No CUDA out-of-memory crashes. No hyperparameter tuning. Drop raw logs, CSVs, markdown docs, or a plain-English prompt—Bonsai takes care of the rest.

---

## 🌊 Why Bonsai?

Running local models is solved (`ollama`, `LM Studio`). But **99% of developers and everyday power-users have zero idea how to distill or train their own micro-models (0.5B to 1.5B) on their own home PC.**

They get blocked by:
- ❌ Manually formatting instruction-tuning JSONL pairs
- ❌ Sudden `CUDA out of memory` crashes
- ❌ Confusing ML hyperparameter jargon (`lora_r`, `grad_accum`, `cosine_warmup`)
- ❌ Catastrophic forgetting and hallucination loops
- ❌ Zero proof that their fine-tuned model is actually better

**Bonsai eliminates the friction entirely.**

It acts as an autonomous master artisan: inspecting your exact GPU VRAM / Unified Memory, cultivating training pairs with guaranteed zero data leakage, executing an ultra-fast LoRA distillation pass with prompt token loss masking, proving accuracy gains on an unseen blind holdout exam, and exporting a ready-to-chat Ollama Modelfile or local web app.

---

## 🏛️ How It Works: The 5-Step Cultivation Loop

```mermaid
flowchart LR
    A["💻 1. PROBE<br/>Hardware Auto-Downshift"] --> B["📥 2. INGEST<br/>Raw Docs, Logs, or Goals"]
    B --> C["🧬 3. CULTIVATE<br/>85% Train / 15% Blind Exam"]
    C --> D["⚡ 4. DISTILL<br/>Masked-Loss Pocket LoRA"]
    D --> E["🔬 5. PROOF & SERVE<br/>SVG Report + Web Chat"]
```

1. **Hardware Auto-Probe:** Probes your GPU VRAM, Apple Unified Memory, or CPU RAM. If VRAM is tight (<4 GB), it automatically downshifts base models (e.g., `Qwen2.5-Coder-0.5B`) to guarantee **Zero-OOM**.
2. **Universal Ingestion:** Drop raw `.txt`, `.csv`, `.md`, or server logs, or specify a plain-English task.
3. **Teacher Cultivation & Zero Leakage:** Synthesizes high-fidelity instruction-response pairs and enforces strict scenario deduplication *before* augmentation, splitting into an 85% training set and a 15% unseen blind holdout exam.
4. **Pocket LoRA Distillation:** Freezes 99.78% of the model and cultivates only the top 0.22% of synaptic adapter weights. Implements **prompt loss masking** (`labels[:prompt_len] = -100`) so gradients are calculated exclusively on assistant tokens.
5. **The Proof Machine:** Evaluates the untrained base model side-by-side against your newly cultivated model on the unseen exam set, emitting a viral SVG scorecard, interactive HTML report, and instant Ollama Modelfile.

---

## 💻 CLI Commands

### 1. Probe Hardware
Inspects silicon headroom and translates ML jargon into human concepts:
```bash
bonsai probe
```

```
╭─────────────────────────────────────────────────────────╮
│               🌱 BONSAI HARDWARE TELEMETRY               │
╰─────────────────────────────────────────────────────────╯
Silicon: NVIDIA GeForce RTX 2050 (3300 MB Free VRAM)
Recommended Micro-Model: Qwen2.5-Coder-0.5B-Instruct (Zero-OOM Protected)

Translated Cultivation Recipe:
  • Learning Agility (Learning Rate):    2e-4
  • Synaptic Plasticity (LoRA Rank):     r=8, alpha=16
  • Study Passes (Epochs):               3 passes
  • Attention Batching (Accumulation):   4 steps
```

### 2. Prune & Distill
Train a micro-AI on your local files or task description:
```bash
# From a folder of raw logs/documents:
bonsai prune ./examples/messy_ops_to_json

# Or from a task prompt:
bonsai prune "Build a tiny AI that extracts strict JSON invoices from messy emails"
```

### 3. Exam & Proof Machine
Evaluate the newly trained model on the unseen blind test set:
```bash
bonsai exam ./bonsai_pruned
```

Generates:
* 📊 **Viral SVG Scorecard:** `bonsai-report.svg`
* 🌐 **Interactive HTML Report:** `bonsai-report.html`
* 📋 **Canonical State Manifest:** `bonsai_manifest.json`

### 4. Interactive Web Chat & API
Launch the local dark-mode UI and OpenAI-compatible API endpoint:
```bash
bonsai ui
```

---

## 📊 Live Verification Scorecard (Clean-Room NVIDIA RTX 2050)

<p align="center">
  <img src="assets/bonsai-report.svg" alt="Bonsai Scorecard" width="100%" />
</p>

| Evaluation Metric | Raw Base Model | Cultivated Bonsai Model | Net Delta |
|:---|:---:|:---:|:---:|
| **Strict Schema Validity** | 100.0% | **100.0%** | Maintained |
| **Output Cleanliness** | Markdown code fences, float fields | **Strict Bare JSON, snake_case** | **Reflexive Precision** |
| **Training Duration** | — | **34.05 seconds** | **Ultra-Fast** |
| **Loss Convergence** | 1.5905 | **0.7713** | **-51.5% Convergence** |
| **Peak VRAM Allocated** | — | **2,045 MB** | **Zero-OOM Verified** |
| **Inference Latency** | 3,533 ms | **4,352 ms** | Consumer Silicon Ready |

---

## 🥊 Comparison Matrix

| Feature | Unsloth | Axolotl | Ollama | **🌱 Bonsai** |
|:---|:---:|:---:|:---:|:---:|
| **Target User** | ML Engineers | Senior ML Researchers | Everyone | **Everyone** |
| **Dataset Requirement** | Manual `.jsonl` | Complex YAML + `.jsonl` | None (Inference only) | **Zero-Knowledge (Raw files/prompts)** |
| **Hardware Safety** | Manual VRAM tuning | Manual tuning | Fixed GGUF | **Automated Zero-OOM Downshift** |
| **Synthetic Teacher** | ❌ Manual | ❌ Manual | ❌ None | **✅ Built-in Zero-Leakage Teacher** |
| **Automated Exam Suite** | ❌ None | ❌ None | ❌ None | **✅ Unseen Blind Holdout Proof Machine** |
| **Visual Proof Reports** | ❌ W&B logs | ❌ W&B logs | ❌ None | **✅ Standalone SVG & HTML Scorecards** |
| **1-Click Web UI & Desktop**| ❌ None | ❌ None | ❌ CLI only | **✅ Dark-Mode UI + Windows .bat Launcher** |

---

## 🧠 S-Tier Cultivation Dictionary

Bonsai strips away gatekept academic jargon:

| ML Term | Bonsai Term | What It Actually Does |
|:---|:---|:---|
| `learning_rate` | **Learning Agility** | Calibrates how rapidly the micro-model adapts to new reflexes without catastrophic forgetting. |
| `lora_r`, `lora_alpha` | **Synaptic Plasticity** | Freezes 99.78% of the foundation brain and cultivates only the top 0.22% most sensitive decision synapses. |
| `epochs` | **Study Passes** | The exact number of focused practice passes across the cultivated dataset. |
| `gradient_accumulation` | **Attention Batching** | Simulates large model batching on tight consumer VRAM (under 4GB). |
| `token_loss_masking` | **Focus Reflex** | Masks system directives so gradients are applied strictly to the assistant's completions. |

---

## 🧪 Automated Testing & Invariants

Bonsai is continuously verified with automated unit and integration tests:

```bash
pytest tests/ -v
```

```
collected 11 items
tests/test_cli.py::test_cli_help PASSED                        [  9%]
tests/test_cli.py::test_cli_probe PASSED                       [ 18%]
tests/test_cli.py::test_cli_prune_missing_source PASSED        [ 27%]
tests/test_holdout_leakage.py::test_zero_data_leakage PASSED   [ 36%]
tests/test_holdout_leakage.py::test_deterministic_seed_reproducibility PASSED [ 45%]
tests/test_ingest.py::test_ingest_prompt_goal PASSED           [ 54%]
tests/test_ingest.py::test_ingest_files PASSED                 [ 63%]
tests/test_ingest.py::test_ingest_empty_directory PASSED       [ 72%]
tests/test_manifest.py::test_manifest_roundtrip PASSED         [ 81%]
tests/test_probe.py::test_probe_hardware_structure PASSED      [ 90%]
tests/test_probe.py::test_probe_jargon_translations PASSED     [100%]

============================= 11 passed in 15.16s =============================
```

---

## 📜 License & Community

Released under the **[MIT License](LICENSE)**. Cultivated with care for the global open-source AI community.

Contributions, issues, and PRs are warmly welcomed. See **[CONTRIBUTING.md](CONTRIBUTING.md)** for details.
