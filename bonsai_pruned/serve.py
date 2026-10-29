# Standalone Bonsai Local Runner
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

BASE_MODEL = r"D:\bonsai\models\Qwen2.5-Coder-0.5B-Instruct"
ADAPTER_PATH = r"D:\bonsai\bonsai_pruned\adapter"
SYSTEM_DIRECTIVE = r"""You are Bonsai Micro-AI, an ultra-fast pocket-sized AI cultivated locally on consumer silicon using the Bonsai distillation engine. You specialize in deterministic, strict-schema execution."""

print("🌱 Loading Bonsai Micro-AI...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype, trust_remote_code=True).to(device)
model = PeftModel.from_pretrained(base, ADAPTER_PATH).to(device)
model.eval()

def query(prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_DIRECTIVE},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

if __name__ == "__main__":
    print("✨ Bonsai ready! Type your prompt (or 'exit' to quit):\n")
    while True:
        try:
            inp = input("Query > ")
            if inp.lower() in ("exit", "quit"):
                break
            print("\nResponse:\n", query(inp), "\n")
        except KeyboardInterrupt:
            break
