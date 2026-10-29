"""
Export & Deployment Engine for Bonsai.
Generates Ollama Modelfiles and zero-dependency local inference servers.
"""

from pathlib import Path
from typing import Dict, Any
from bonsai.core.constants import DEFAULT_SYSTEM_DIRECTIVE

def export_ollama_modelfile(
    base_model: str,
    adapter_path: str,
    output_path: str = "./Modelfile",
    system_prompt: str = DEFAULT_SYSTEM_DIRECTIVE
) -> Path:
    """Creates an Ollama Modelfile ready for instant `ollama create bonsai-mini -f Modelfile`."""
    file_path = Path(output_path)
    content = f"""# Bonsai Cultivated Model
# Exported by Bonsai v0.1

FROM {base_model}
ADAPTER {adapter_path}

# System Directive
SYSTEM \"\"\"{system_prompt}\"\"\"

# Execution Parameters
PARAMETER temperature 0.1
PARAMETER top_p 0.95
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
"""
    file_path.write_text(content, encoding="utf-8")
    return file_path

def generate_local_runner(
    base_model: str,
    adapter_path: str,
    output_path: str = "./serve.py",
    system_prompt: str = DEFAULT_SYSTEM_DIRECTIVE
) -> Path:
    """Generates a standalone Python inference runner for immediate testing."""
    file_path = Path(output_path)
    content = f'''# Standalone Bonsai Local Runner
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = r"{base_model}"
ADAPTER_PATH = r"{adapter_path}"
SYSTEM_DIRECTIVE = r"""{system_prompt}"""

print("🌱 Loading Bonsai Micro-AI...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype, trust_remote_code=True).to(device)
model = PeftModel.from_pretrained(base, ADAPTER_PATH).to(device)
model.eval()

def query(prompt: str) -> str:
    messages = [
        {{"role": "system", "content": SYSTEM_DIRECTIVE}},
        {{"role": "user", "content": prompt}}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

if __name__ == "__main__":
    print("✨ Bonsai ready! Type your prompt (or 'exit' to quit):\\n")
    while True:
        try:
            inp = input("Query > ")
            if inp.lower() in ("exit", "quit"):
                break
            print("\\nResponse:\\n", query(inp), "\\n")
        except KeyboardInterrupt:
            break
'''
    file_path.write_text(content, encoding="utf-8")
    return file_path
