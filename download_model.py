import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import time
from pathlib import Path
import requests

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
TARGET_DIR = Path("D:/bonsai/models/Qwen2.5-Coder-0.5B-Instruct")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json"
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

print(f"🌱 Downloading {MODEL_ID} to {TARGET_DIR}...")

for fname in FILES:
    dest = TARGET_DIR / fname
    if dest.exists() and dest.stat().st_size > 0:
        if fname != "model.safetensors" or dest.stat().st_size >= 900_000_000:
            print(f"✓ {fname} already cached ({round(dest.stat().st_size / 1024, 1)} KB)")
            continue

    url = f"https://huggingface.co/{MODEL_ID}/resolve/main/{fname}"
    print(f"Downloading {fname} from {url}...")
    
    r = requests.get(url, stream=True, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        print(f"Failed to fetch {fname}: {r.status_code}")
        continue
        
    total_size = int(r.headers.get("content-length", 0))
    start = time.time()
    downloaded = 0
    
    temp_dest = dest.with_suffix(".tmp")
    with open(temp_dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                pct = (downloaded / total_size * 100) if total_size else 0
                mb = downloaded / (1024 * 1024)
                rate = mb / max(time.time() - start, 0.01)
                print(f"\r  {fname}: {mb:.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%) @ {rate:.1f} MB/s", end="", flush=True)
                
    temp_dest.replace(dest)
    print(f"\n✓ Completed {fname} in {round(time.time() - start, 1)}s")

print("✨ All model files successfully cached!")
