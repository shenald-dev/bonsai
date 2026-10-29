"""
Bonsai Web UI & Live Inference Server.
Single-process, zero-reload GPU server with a dark-mode interactive web chat interface.
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import os
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from bonsai.core.constants import get_resolved_model_path, DEFAULT_SYSTEM_DIRECTIVE
from bonsai.core.manifest import load_manifest

PORT = 7860

# Resolve paths dynamically via manifest or default resolver
try:
    manifest = load_manifest("D:/bonsai/bonsai_pruned")
    BASE_MODEL_DIR = manifest.base_model
    ADAPTER_DIR = manifest.adapter_dir
except Exception:
    BASE_MODEL_DIR = get_resolved_model_path()
    ADAPTER_DIR = "D:/bonsai/bonsai_pruned/adapter"

print("==================================================================")
print("🌱 BONSAI LIVE CHAT SERVER")
print("Loading fine-tuned micro-AI into NVIDIA GPU VRAM (one-time load)...")
print("==================================================================")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# 1. Load Model into GPU once
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    torch_dtype=dtype,
    trust_remote_code=True,
    low_cpu_mem_usage=True
).to(device)

model = PeftModel.from_pretrained(base_model, ADAPTER_DIR).to(device)
model.eval()

vram_mb = 0.0
if device == "cuda":
    vram_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)

print(f"✓ Model loaded successfully on {device.upper()}! VRAM allocated: {vram_mb} MB")
print(f"✓ Web UI ready at: http://localhost:{PORT}")
print("==================================================================\n")

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Bonsai Micro-AI • Live Local Chat</title>
  <style>
    :root {
      --bg: #090e0b;
      --card-bg: #111a14;
      --border: #1f3325;
      --accent: #10b981;
      --accent-light: #34d399;
      --text: #e2e8f0;
      --text-dim: #94a3b8;
      --msg-user: #132c20;
      --msg-ai: #0e1712;
      --code-bg: #060a08;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: var(--card-bg);
      border-bottom: 1px solid var(--border);
      padding: 16px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .logo-group {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .logo-icon {
      font-size: 26px;
    }
    .logo-title {
      font-size: 18px;
      font-weight: 700;
      color: var(--accent-light);
      letter-spacing: 0.5px;
    }
    .logo-sub {
      font-size: 12px;
      color: var(--text-dim);
    }
    .hardware-badge {
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.3);
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 12px;
      color: var(--accent-light);
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 600;
    }
    .pulse-dot {
      width: 8px;
      height: 8px;
      background: var(--accent);
      border-radius: 50%;
      box-shadow: 0 0 8px var(--accent);
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0% { opacity: 0.4; }
      50% { opacity: 1; }
      100% { opacity: 0.4; }
    }
    main {
      flex: 1;
      display: flex;
      flex-direction: column;
      max-width: 900px;
      width: 100%;
      margin: 0 auto;
      padding: 20px;
      overflow: hidden;
    }
    #chat-container {
      flex: 1;
      overflow-y: auto;
      padding-right: 8px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    #chat-container::-webkit-scrollbar {
      width: 6px;
    }
    #chat-container::-webkit-scrollbar-thumb {
      background: var(--border);
      border-radius: 3px;
    }
    .message {
      max-width: 85%;
      padding: 14px 18px;
      border-radius: 12px;
      line-height: 1.5;
      font-size: 14.5px;
      animation: fadeIn 0.25s ease-out;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .message.user {
      align-self: flex-end;
      background: var(--msg-user);
      border: 1px solid #1c4a33;
      border-bottom-right-radius: 2px;
    }
    .message.ai {
      align-self: flex-start;
      background: var(--msg-ai);
      border: 1px solid var(--border);
      border-bottom-left-radius: 2px;
    }
    .message pre {
      background: var(--code-bg);
      border: 1px solid #1a2a20;
      padding: 12px;
      border-radius: 8px;
      overflow-x: auto;
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      font-size: 13px;
      margin: 8px 0;
      color: #a7f3d0;
    }
    .meta-tag {
      font-size: 11px;
      color: var(--text-dim);
      margin-top: 6px;
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .quick-chips {
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding: 10px 0;
      margin-top: 10px;
    }
    .quick-chips::-webkit-scrollbar { display: none; }
    .chip {
      background: var(--card-bg);
      border: 1px solid var(--border);
      color: var(--text-dim);
      padding: 6px 12px;
      border-radius: 16px;
      font-size: 12px;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.2s;
    }
    .chip:hover {
      background: rgba(16, 185, 129, 0.15);
      border-color: var(--accent);
      color: var(--accent-light);
    }
    .input-bar {
      margin-top: 10px;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 8px 12px;
      display: flex;
      align-items: flex-end;
      gap: 10px;
    }
    .input-bar:focus-within {
      border-color: var(--accent);
      box-shadow: 0 0 10px rgba(16, 185, 129, 0.2);
    }
    textarea {
      flex: 1;
      background: transparent;
      border: none;
      color: var(--text);
      font-size: 14.5px;
      font-family: inherit;
      resize: none;
      outline: none;
      max-height: 140px;
      line-height: 1.4;
      padding: 6px 4px;
    }
    textarea::placeholder {
      color: #64748b;
    }
    button#send-btn {
      background: var(--accent);
      color: #042416;
      border: none;
      border-radius: 8px;
      padding: 8px 16px;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: opacity 0.2s;
    }
    button#send-btn:hover {
      opacity: 0.9;
    }
    button#send-btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
  </style>
</head>
<body>
  <header>
    <div class="logo-group">
      <div class="logo-icon">🌱</div>
      <div>
        <div class="logo-title">BONSAI MICRO-AI</div>
        <div class="logo-sub">Cultivated Qwen2.5-Coder 0.5B • Consumer Silicon</div>
      </div>
    </div>
    <div class="hardware-badge">
      <div class="pulse-dot"></div>
      <span id="hw-info">GPU: RTX 2050 (Live)</span>
    </div>
  </header>

  <main>
    <div id="chat-container">
      <div class="message ai">
        <strong>🌱 Bonsai Micro-AI:</strong><br>
        Hello! I am your cultivated micro-AI, fine-tuned locally on your RTX 2050 GPU. You can chat with me freely, or click any sample below to test my deterministic incident parsing and JSON extraction!
      </div>
    </div>

    <div class="quick-chips">
      <div class="chip" onclick="fillPrompt('CRITICAL: db-cluster pg_bouncer pool exhausted on worker-node-04. 2500/2500 active connections.')">🚨 DB Pool Crash</div>
      <div class="chip" onclick="fillPrompt('FATAL: billing-worker pod terminated by Linux kernel OOM-Killer. Memory exceeded 2048Mi.')">💥 Pod OOM-Killer</div>
      <div class="chip" onclick="fillPrompt('WARN: redis-cache node-02 observed high eviction rate. Cache hit ratio dropped to 71.2%.')">⚠️ Redis Storm</div>
      <div class="chip" onclick="fillPrompt('CRITICAL: auth-api JWT signature validation latency escalated to 4500ms due to key rotation contention.')">🔒 JWT Failure</div>
      <div class="chip" onclick="fillPrompt('Who are you and what makes you special compared to giant foundation models?')">💬 Tell me about yourself</div>
    </div>

    <div class="input-bar">
      <textarea id="prompt-input" rows="1" placeholder="Type a message or paste a raw incident log (Enter to send, Shift+Enter for new line)..."></textarea>
      <button id="send-btn" onclick="sendMessage()">Send</button>
    </div>
  </main>

  <script>
    const chatContainer = document.getElementById('chat-container');
    const promptInput = document.getElementById('prompt-input');
    const sendBtn = document.getElementById('send-btn');

    promptInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    promptInput.addEventListener('input', () => {
      promptInput.style.height = 'auto';
      promptInput.style.height = Math.min(promptInput.scrollHeight, 140) + 'px';
    });

    function fillPrompt(text) {
      promptInput.value = text;
      promptInput.focus();
      promptInput.style.height = 'auto';
      promptInput.style.height = Math.min(promptInput.scrollHeight, 140) + 'px';
    }

    function appendMessage(role, text, meta) {
      const msgDiv = document.createElement('div');
      msgDiv.className = `message ${role}`;

      // Check if text contains JSON or code
      let formatted = text;
      if (text.includes('{') && text.includes('}')) {
        try {
          const match = text.match(/\{[\s\S]*\}/);
          if (match) {
            const parsed = JSON.parse(match[0]);
            const pretty = JSON.stringify(parsed, null, 2);
            formatted = text.replace(match[0], `<pre><code>${pretty}</code></pre>`);
          }
        } catch (e) {}
      } else {
        formatted = text.replace(/\\n/g, '<br>');
      }

      msgDiv.innerHTML = formatted;

      if (meta) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'meta-tag';
        metaDiv.innerHTML = `⚡ ${meta.latency_ms} ms • ${meta.tokens_sec} tok/s • Memory: ${meta.vram_mb} MB`;
        msgDiv.appendChild(metaDiv);
      }

      chatContainer.appendChild(msgDiv);
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    async function sendMessage() {
      const prompt = promptInput.value.trim();
      if (!prompt) return;

      appendMessage('user', prompt);
      promptInput.value = '';
      promptInput.style.height = 'auto';
      sendBtn.disabled = true;

      // Temporary typing indicator
      const typingDiv = document.createElement('div');
      typingDiv.className = 'message ai';
      typingDiv.id = 'typing-indicator';
      typingDiv.innerHTML = '<em>🌱 Cultivating response on GPU...</em>';
      chatContainer.appendChild(typingDiv);
      chatContainer.scrollTop = chatContainer.scrollHeight;

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: prompt })
        });
        const data = await res.json();
        typingDiv.remove();

        appendMessage('ai', data.response, {
          latency_ms: data.latency_ms,
          tokens_sec: data.tokens_sec,
          vram_mb: data.vram_mb
        });
      } catch (err) {
        typingDiv.remove();
        appendMessage('ai', '❌ Error connecting to local Bonsai server: ' + err.message);
      } finally {
        sendBtn.disabled = false;
        promptInput.focus();
      }
    }
  </script>
</body>
</html>
"""

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class BonsaiHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
        elif self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            curr_vram = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if device == "cuda" else 0.0
            data = {"status": "ready", "device": device, "vram_mb": curr_vram, "model": BASE_MODEL_DIR}
            self.wfile.write(json.dumps(data).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/chat":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")
            data = json.loads(body)
            prompt = data.get("prompt", "")

            # Run inference
            start_time = time.perf_counter()
            system_directive = (
                "You are Bonsai Micro-AI, an ultra-fast pocket-sized AI cultivated locally on consumer silicon "
                "using the Bonsai distillation engine. You specialize in zero-latency incident extraction, structured JSON "
                "parsing, and deterministic reasoning. When asked who you are or what model you are, identify yourself "
                "proudly as Bonsai Micro-AI running locally on this PC."
            )
            messages = [
                {"role": "system", "content": system_directive},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            elapsed = time.perf_counter() - start_time
            gen = out[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(gen, skip_special_tokens=True).strip()

            num_tokens = len(gen)
            latency_ms = round(elapsed * 1000, 1)
            tokens_sec = round(num_tokens / elapsed, 1) if elapsed > 0 else 0.0
            curr_vram = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if device == "cuda" else 0.0

            res_payload = {
                "response": response,
                "latency_ms": latency_ms,
                "tokens_sec": tokens_sec,
                "vram_mb": curr_vram
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(res_payload).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Clean logging
        pass

def run_server():
    server = ThreadedHTTPServer(("127.0.0.1", PORT), BonsaiHandler)
    # Open browser automatically
    try:
        webbrowser.open(f"http://localhost:{PORT}")
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🌱 Shutting down Bonsai server cleanly...")
        server.server_close()

if __name__ == "__main__":
    run_server()
