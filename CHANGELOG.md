# Changelog

All notable changes to the "Bonsai" project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-01

### Added
- **LoRA Adapter Merging Engine (`bonsai merge`):** Added `BonsaiMerger` to fuse trained PEFT synaptic adapter weights directly into the base foundation model using `merge_and_unload()`. Saves standalone, unquantized weights ready for zero-overhead inference or direct GGUF/llama.cpp quantization.
- **Universal Multi-Archetype Teacher Engine:** Upgraded `TeacherEngine` with multi-domain offline programmatic synthesizers:
  - Financial, invoice, and receipt extraction schemas
  - Customer support ticket triage and urgency classification
  - Natural language to SQL query generation
  - Tabular and CSV record attribute extraction
  - Python and JavaScript/TypeScript function and docstring extraction
  - Domain-adaptive instruction prompt augmentation
- **OpenAI-Compatible Local API Server:**
  - Added standards-compliant `/v1/chat/completions` endpoint for direct integration with Cursor, VS Code, LangChain, LlamaIndex, or curl.
  - Added `/v1/models` and `/health` diagnostic endpoints.
  - Added configurable `--port`, `--host`, `--target`, and `--no-browser` flags to `bonsai ui`.
- **Generalized Multi-Metric Proof Machine:**
  - Upgraded `BonsaiExam` to evaluate both strict JSON schemas and free-form text or code.
  - Computes exact match, token-level F1 semantic overlap, latency deltas, and throughput.
- **Cross-Platform Parity & Dynamic Resolution:**
  - Added `run_bonsai.sh` for Linux and macOS execution.
  - Removed all hardcoded Windows drive letters and paths from `web_server.py`, `download_model.py`, and `run_bonsai.bat`.
  - Added lazy-loading model architecture in `BonsaiServer` to eliminate import-time side effects and VRAM allocation.
- **Expanded Test Suite:** Added 17 new comprehensive test cases (now 28 tests passing) covering model merging, multi-archetype synthesis, OpenAI API compliance, and multi-metric evaluation.

## [0.1.0] - 2026-10-29

### Added
- **Initial Release:** Core architecture for pocket-sized micro-model distillation.
- **Hardware Telemetry Probe:** Probes silicon (CUDA, Apple Silicon MPS, CPU) and auto-downshifts micro-models (Qwen2.5-Coder-0.5B / 1.5B) to guarantee Zero-OOM.
- **Zero-Leakage Holdout Separation:** 85% train vs. 15% blind exam holdout split with Jaccard-similarity verification.
- **Masked-Loss Pocket LoRA Distillation:** Fast local LoRA fine-tuning with prompt token loss masking (`labels[:prompt_len] = -100`).
- **Ollama Modelfile Exporter:** Automated generation of `Modelfile` and standalone `serve.py` runners.
- **Visual Scorecard Reporting:** Terminal ASCII scorecard, standalone SVG badge, and interactive HTML report.
