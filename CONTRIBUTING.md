# Contributing to Bonsai 🌱

Thank you for your interest in contributing to **Bonsai**! 
Our goal is to make local micro-model distillation accessible to every developer on consumer hardware without requiring a PhD in machine learning.

---

## 🛠️ Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/bonsai.git
   cd bonsai
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies in editable mode:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run the test suite:**
   ```bash
   pytest tests/ -v
   ```

---

## 🏛️ Architecture Principles

When contributing code, adhere to the 4 Bonsai Core Invariants:

1. **Zero Mock Policy:** Never fake progress bars, telemetry numbers, or benchmark gains. All metrics must come from genuine model forward/backward passes.
2. **Zero-OOM Safety:** Hardware probing must always protect the consumer user's machine from Out-Of-Memory crashes. If VRAM is tight, downshift gracefully.
3. **Zero Data Leakage:** Always split raw scenarios *before* data augmentation to guarantee 100% blind holdout exams.
4. **Chat Template Parity:** Ensure the system directive, chat format, and prompt loss masking remain synchronized across `trainer`, `exam`, `serve`, and `web_server`.

---

## 🧪 Submitting a Pull Request

1. Create a feature branch: `git checkout -b feature/my-enhancement`.
2. Write unit tests in `tests/` covering your changes.
3. Verify all tests pass: `pytest tests/ -v`.
4. Ensure code formatting is clean.
5. Open a Pull Request with a clear description of the problem solved.

Thank you for cultivating pocket-sized intelligences with us! 🌿
