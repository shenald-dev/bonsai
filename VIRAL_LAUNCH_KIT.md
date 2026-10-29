# 🚀 Bonsai Viral Launch Kit: Go Everywhere

> **Repo:** https://github.com/shenald-dev/bonsai  
> **Tagline:** The art of cultivating massive foundation models into pocket-sized intelligences on consumer silicon.  
> **One-Liner:** `uvx bonsai prune ./my_data`

---

## 1. 🟠 Hacker News (Show HN)

**Target:** https://news.ycombinator.com/submit  
**Title:** Show HN: Bonsai – Distill 0.5B–1.5B micro-models locally on a laptop in 35s  
**Text / Body:**

```
Hey HN,

Running local models is solved (Ollama, LM Studio). But fine-tuning or distilling micro-models (0.5B to 1.5B parameters) remains out of reach for 99% of developers because of PyTorch CUDA OOM crashes, dataset formatting, and hyperparameter guessing.

We built Bonsai (https://github.com/shenald-dev/bonsai) to make local distillation zero-knowledge and zero-config on everyday consumer silicon (laptops with 4GB VRAM, Apple Silicon, or CPU).

You just drop in raw text files, server logs, or a plain-English goal ("Build a micro-AI that extracts strict JSON incidents from messy logs").

Here is what happens under the hood:
1. Hardware Auto-Probe: Inspects exact VRAM headroom and automatically downshifts models (e.g. Qwen2.5-Coder-0.5B) and batch sizes to guarantee Zero-OOM.
2. Zero-Leakage Synthetic Teacher: Synthesizes instruction pairs and partitions raw scenarios before template augmentation, enforcing a strict 85% train vs 15% blind holdout exam (Jaccard similarity = 0.0).
3. Masked-Loss Pocket LoRA: Only trains 0.22% of weights (LoRA r=8) with prompt token loss masking (labels[:prompt_len] = -100) so gradients are applied strictly to the assistant's completions. On an RTX 2050 (4GB), 3 epochs finished in 34.05s, dropping loss from 1.59 to 0.77 (-51.5%).
4. The Proof Machine: Runs the newly trained adapter side-by-side against the raw base model on unseen test questions, outputting SVG/HTML report cards.
5. Export: Emits an Ollama Modelfile, standalone runner, and local dark-mode Web UI.

It's 100% open-source (MIT). We'd love feedback on your hardware configurations and edge-case workloads.

GitHub: https://github.com/shenald-dev/bonsai
```

---

## 2. 🔴 Reddit (r/LocalLLaMA & r/MachineLearning)

**Target Subreddits:**
- r/LocalLLaMA (Primary)
- r/MachineLearning
- r/Python
- r/OpenSource

**Title:** [Project] Bonsai: Zero-Knowledge Micro-Model Distillation on 4GB Consumer GPUs (34s Train Time, Zero-OOM, Proof Machine)

**Post Body:**

```markdown
Hey r/LocalLLaMA!

Most people think you need an A100 cluster or deep ML engineering experience to distill foundation models. We wanted to see how far we could push a zero-config, consumer-first distillation pipeline that anyone can run with a single command.

We open-sourced **Bonsai**: https://github.com/shenald-dev/bonsai

### The Problem
Fine-tuning QLoRA via Unsloth/Axolotl is awesome, but everyday developers get tripped up by:
- Manual JSONL dataset construction
- Unmasked prompt loss (model memorizes system prompts instead of output reflexes)
- Accidental data leakage between train/test sets
- CUDA out-of-memory crashes on 4GB-6GB laptop GPUs
- Having no quantitative proof that their fine-tune actually improved over the base model

### How Bonsai Works (`uvx bonsai prune ./data`)
1. **Hardware Downshift Guard:** Probes your silicon and sets the batch size and micro-model (defaulting to Qwen2.5-Coder-0.5B/1.5B) so it fits in <2.1 GB VRAM.
2. **Zero-Leakage Scenario Partitioner:** De-duplicates input scenarios before augmentation so the 15% blind holdout exam is truly unseen.
3. **Prompt Loss Masking:** Masks input tokens (`labels[:prompt_len] = -100`) so loss is calculated purely on the completion.
4. **Side-by-Side Holdout Evaluation:** Executes baseline passes under `model.disable_adapter()` with CUDA memory cache purges, then compares schema validity, token speed, and latency against the pruned model.
5. **Instant Export:** Generates an Ollama Modelfile, standalone Python runner, and interactive dark-mode Web UI.

### Live Clean-Room Benchmark (Laptop RTX 2050 - 4GB VRAM):
- Dataset: 24 cultivated samples, 2 blind exam questions
- Training Time: 34.05 seconds
- Loss: 1.5905 -> 0.7713 (-51.5% drop)
- Peak VRAM: 2,045 MB
- Model: Qwen2.5-Coder-0.5B-Instruct LoRA (r=8, alpha=16)

Everything is tested with pytest (11/11 tests passing) and open-sourced under MIT.

Check it out on GitHub: https://github.com/shenald-dev/bonsai
Looking forward to your thoughts and PRs!
```

---

## 3. 🐦 X / Twitter Viral Thread

**Tweet 1 (The Hook):**
> Most developers think you need an A100 cluster to distill AI models.
> 
> Today, I'm open-sourcing Bonsai 🌱
> 
> Distill foundation models into pocket-sized 0.5B–1.5B intelligences on your home laptop in 35 seconds. Zero-OOM. Zero ML jargon.
> 
> `uvx bonsai prune ./data`
> 
> 🧵👇 (1/6)
> [Attach `assets/bonsai-hero-banner.png`]

**Tweet 2 (The Friction):**
> Running local models with Ollama is easy.
> 
> But fine-tuning is gatekept by:
> ❌ Formatting raw data into JSONL
> ❌ Random CUDA out-of-memory crashes
> ❌ Hyperparameter anxiety (LoRA rank, warmup, cosine decay)
> ❌ Zero proof the fine-tune actually works
> 
> Bonsai fixes this completely. (2/6)

**Tweet 3 (The 5-Step Loop):**
> How Bonsai works under the hood:
> 
> 1. Auto-probes your VRAM (Zero-OOM safety guard)
> 2. Synthesizes 85% train + 15% unseen blind exam
> 3. Executes masked-loss LoRA pruning
> 4. Tests base vs. cultivated model side-by-side
> 5. Exports 1-click Ollama Modelfile & Web UI (3/6)

**Tweet 4 (The Real Hardware Numbers):**
> Clean-room test on an everyday laptop RTX 2050 (4GB VRAM):
> 
> • Base: Qwen2.5-Coder-0.5B
> • Peak VRAM: 2,045 MB (under the 3.3GB limit)
> • Training time: 34.05s
> • Loss: 1.5905 ➔ 0.7713 (-51.5% drop)
> • Output: Strict bare JSON with zero markdown wrapping (4/6)
> [Attach `assets/bonsai-report.svg`]

**Tweet 5 (The Cult UX):**
> We also replaced academic jargon with human concepts:
> 
> • `learning_rate` ➔ Learning Agility
> • `lora_r` ➔ Synaptic Plasticity
> • `epochs` ➔ Study Passes
> • `grad_accum` ➔ Attention Batching
> 
> Built for builders, not PhD defense panels. (5/6)

**Tweet 6 (CTA):**
> Bonsai v0.1 is 100% open-source under MIT.
> 
> ⭐ Star the repo: https://github.com/shenald-dev/bonsai
> 💻 Try it: `uvx bonsai prune ./my_data`
> 
> If you find this useful, a RT on the first tweet helps more devs run local micro-models! 🌿 (6/6)

---

## 4. 🐱 Product Hunt / Indie Hackers Launch

**Tagline:** Cultivate foundation models into pocket-sized intelligences on consumer PCs.  
**Short Description:** A zero-config CLI that turns messy files into high-speed 0.5B–1.5B micro-AIs on 4GB laptops in 35 seconds. With built-in zero-OOM protection and holdout testing.  
**Pricing:** 100% Free & Open Source (MIT)  

---

## 5. 💼 LinkedIn Tech Architecture Post

```markdown
Why run a 70B parameter model in the cloud for simple JSON extraction when a 0.5B micro-model running locally in 970MB of RAM can do it faster, privately, and at zero cost?

Today, I’m releasing Bonsai 🌱 — a zero-knowledge local AI distillation tool for consumer silicon.

Key Innovations:
1. Automated VRAM Probing & Zero-OOM Downshifting
2. Mathematically Verified Zero Data-Leakage Teacher Synthesis
3. Token-Level Loss Masking (focuses gradients exclusively on completions)
4. Built-In Automated Holdout Testing & Proof Scorecards
5. 1-Click Export to Ollama, Python runners, and local Web UI

Clean-room benchmark on a modest 4GB GPU:
⚡ 34.05s training time
📉 -51.5% loss reduction (1.59 -> 0.77)
💾 2,045 MB peak VRAM

Check out the architecture and code on GitHub:
https://github.com/shenald-dev/bonsai

#AI #MachineLearning #OpenSource #LocalAI #LLM #SoftwareEngineering #Python #PyTorch
```
