"""
Bonsai Web UI & OpenAI-Compatible Inference Server.
Single-process, zero-reload local server supporting both an interactive dark-mode web chat
and standard OpenAI-compatible API endpoints (/v1/chat/completions, /v1/models).
"""

import json
import os
import sys
import time
import uuid
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Dict, Any, Optional

from bonsai.core.constants import get_resolved_model_path, DEFAULT_SYSTEM_DIRECTIVE
from bonsai.core.manifest import load_manifest


# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Bonsai Micro-AI • Live Local Chat & API</title>
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
      padding: 14px 24px;
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
    .badges {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .badge {
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
      overflow: hidden;
    }
    #chat-container {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .message {
      max-width: 80%;
      padding: 14px 18px;
      border-radius: 12px;
      line-height: 1.5;
      font-size: 14px;
      position: relative;
      word-break: break-word;
    }
    .message.user {
      align-self: flex-end;
      background: var(--msg-user);
      border: 1px solid var(--border);
      border-bottom-right-radius: 2px;
    }
    .message.ai {
      align-self: flex-start;
      background: var(--msg-ai);
      border: 1px solid var(--border);
      border-bottom-left-radius: 2px;
    }
    .telemetry-tag {
      font-size: 11px;
      color: var(--accent);
      margin-top: 8px;
      display: flex;
      gap: 12px;
      opacity: 0.85;
      border-top: 1px dashed rgba(16, 185, 129, 0.2);
      padding-top: 6px;
    }
    pre {
      background: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
      margin: 8px 0;
      overflow-x: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 13px;
      color: #6ee7b7;
    }
    footer {
      background: var(--card-bg);
      border-top: 1px solid var(--border);
      padding: 16px 24px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .input-row {
      display: flex;
      gap: 12px;
    }
    textarea {
      flex: 1;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 16px;
      color: var(--text);
      font-family: inherit;
      font-size: 14px;
      resize: none;
      height: 48px;
      outline: none;
      transition: border-color 0.2s;
    }
    textarea:focus {
      border-color: var(--accent);
    }
    button.send-btn {
      background: var(--accent);
      color: #064e3b;
      font-weight: 700;
      border: none;
      border-radius: 8px;
      padding: 0 24px;
      cursor: pointer;
      transition: background 0.2s;
      font-size: 14px;
    }
    button.send-btn:hover {
      background: var(--accent-light);
    }
    button.send-btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .quick-chips {
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 2px;
    }
    .chip {
      background: var(--bg);
      border: 1px solid var(--border);
      color: var(--text-dim);
      font-size: 12px;
      padding: 4px 10px;
      border-radius: 14px;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.2s;
    }
    .chip:hover {
      border-color: var(--accent);
      color: var(--text);
    }
  </style>
</head>
<body>
  <header>
    <div class="logo-group">
      <div class="logo-icon">🌱</div>
      <div>
        <div class="logo-title">BONSAI MICRO-AI</div>
        <div class="logo-sub">Ultra-fast pocket intelligence on consumer silicon</div>
      </div>
    </div>
    <div class="badges">
      <div class="badge">
        <div class="pulse-dot"></div>
        <span id="hardware-text">PROBING SILICON...</span>
      </div>
      <div class="badge" title="OpenAI API Compatible Endpoint">
        <span>⚡ OpenAI API: /v1</span>
      </div>
    </div>
  </header>

  <main>
    <div id="chat-container">
      <div class="message ai">
        <strong>Greetings!</strong> I am your locally-cultivated <strong>Bonsai Micro-AI</strong>, operating directly inside your host memory.
        <br/><br/>
        Send me any log snippet, technical problem, or query to verify my responses. You can also connect to me via any OpenAI client at <code>http://localhost:PORT/v1/chat/completions</code>!
      </div>
    </div>
  </main>

  <footer>
    <div class="quick-chips">
      <span class="chip" onclick="fillPrompt('CRITICAL: db-cluster connection pool exhausted on worker-01. Max connections reached.')">💥 DB Pool Exhaustion</span>
      <span class="chip" onclick="fillPrompt('FATAL: billing-worker pod terminated by OOM-Killer. Memory exceeded 2048Mi.')">🛑 Worker OOM Alert</span>
      <span class="chip" onclick="fillPrompt('WARN: redis-cache node-02 high eviction rate. Cache hit ratio dropped to 68%.')">⚠️ Redis Latency Spike</span>
      <span class="chip" onclick="fillPrompt('Who are you and what architecture are you running on?')">🧬 Identity Check</span>
    </div>
    <div class="input-row">
      <textarea id="prompt-input" placeholder="Type a prompt or test query... (Shift+Enter for newline, Enter to send)"></textarea>
      <button id="send-btn" class="send-btn" onclick="sendQuery()">Cultivate</button>
    </div>
  </footer>

  <script>
    const chatContainer = document.getElementById('chat-container');
    const promptInput = document.getElementById('prompt-input');
    const sendBtn = document.getElementById('send-btn');
    const hardwareText = document.getElementById('hardware-text');

    async function updateStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        hardwareText.textContent = `${data.device.toUpperCase()} • VRAM ${data.vram_mb} MB • READY`;
      } catch (e) {
        hardwareText.textContent = 'ONLINE';
      }
    }
    updateStatus();

    promptInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendQuery();
      }
    });

    function fillPrompt(text) {
      promptInput.value = text;
      promptInput.focus();
    }

    function appendMessage(role, text, telemetry = null) {
      const msgDiv = document.createElement('div');
      msgDiv.className = `message ${role}`;

      let formattedText = text;
      if (text.includes('```')) {
        formattedText = text.replace(/```(?:json)?\s*([\\s\\S]*?)\\s*```/g, '<pre><code>$1</code></pre>');
      } else if (text.trim().startsWith('{') && text.trim().endsWith('}')) {
        try {
          const parsed = JSON.parse(text);
          formattedText = '<pre><code>' + JSON.stringify(parsed, null, 2) + '</code></pre>';
        } catch (e) {}
      }

      msgDiv.innerHTML = formattedText;

      if (telemetry) {
        const telDiv = document.createElement('div');
        telDiv.className = 'telemetry-tag';
        telDiv.innerHTML = `<span>⚡ ${telemetry.tokens_sec} tok/s</span><span>⏱ ${telemetry.latency_ms} ms</span><span>💾 ${telemetry.vram_mb} MB VRAM</span>`;
        msgDiv.appendChild(telDiv);
      }

      chatContainer.appendChild(msgDiv);
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    async function sendQuery() {
      const prompt = promptInput.value.trim();
      if (!prompt) return;

      appendMessage('user', prompt);
      promptInput.value = '';
      sendBtn.disabled = true;

      const typingDiv = document.createElement('div');
      typingDiv.className = 'message ai';
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


class BonsaiServer:
    """Encapsulates the PyTorch model runtime and local HTTP / OpenAI server."""

    def __init__(
        self,
        target_path: str = "./bonsai_pruned",
        host: str = "127.0.0.1",
        port: int = 7860
    ):
        self.target_path = Path(target_path).resolve()
        self.host = host
        self.port = port
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.model_identifier = "bonsai-micro-ai"
        self.vram_allocated_mb = 0.0
        self.system_directive = DEFAULT_SYSTEM_DIRECTIVE
        self.httpd: Optional[ThreadedHTTPServer] = None

    def load_model(self):
        """Loads model weights onto device on demand."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # 1. Resolve base model and adapter or merged directory
        base_model_path = None
        adapter_path = None

        if self.target_path.is_file() and self.target_path.name == "bonsai_manifest.json":
            manifest = load_manifest(self.target_path)
            base_model_path = manifest.base_model
            adapter_path = manifest.adapter_dir
        elif (self.target_path / "bonsai_manifest.json").exists():
            manifest = load_manifest(self.target_path / "bonsai_manifest.json")
            base_model_path = manifest.base_model
            adapter_path = manifest.adapter_dir
        elif (self.target_path / "model.safetensors").exists():
            # Merged standalone model directory
            base_model_path = str(self.target_path)
            adapter_path = None
        elif (self.target_path / "adapter" / "adapter_model.safetensors").exists():
            base_model_path = get_resolved_model_path()
            adapter_path = str(self.target_path / "adapter")
        else:
            base_model_path = get_resolved_model_path()
            adapter_path = str(self.target_path / "adapter") if (self.target_path / "adapter").exists() else None

        self.model_identifier = Path(base_model_path).name if Path(base_model_path).exists() else base_model_path

        print(f"🌱 [Bonsai Server] Loading model: {base_model_path}")
        if adapter_path:
            print(f"🌿 [Bonsai Server] Loading adapter: {adapter_path}")

        # Load Tokenizer
        tokenizer_src = adapter_path if adapter_path and (Path(adapter_path) / "tokenizer_config.json").exists() else base_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Base Model
        base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(self.device)

        # Attach LoRA adapter if present
        if adapter_path and Path(adapter_path).exists():
            self.model = PeftModel.from_pretrained(base, adapter_path).to(self.device)
        else:
            self.model = base

        self.model.eval()

        if self.device == "cuda":
            self.vram_allocated_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)

        print(f"✓ Model ready on {self.device.upper()}! VRAM allocated: {self.vram_allocated_mb} MB")

    def generate(self, messages: list, max_tokens: int = 256, temperature: float = 0.0) -> Dict[str, Any]:
        """Runs single conversation completion through the loaded model."""
        import torch

        if not self.model or not self.tokenizer:
            self.load_model()

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        raw_inputs = self.tokenizer(text, return_tensors="pt")
        if hasattr(raw_inputs, "to"):
            inputs = raw_inputs.to(self.device)
        elif isinstance(raw_inputs, dict):
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in raw_inputs.items()}
        else:
            inputs = raw_inputs

        start_time = time.perf_counter()
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            if temperature > 0.0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
            else:
                gen_kwargs["do_sample"] = False

            out = self.model.generate(**inputs, **gen_kwargs)

        elapsed = time.perf_counter() - start_time
        prompt_len = inputs["input_ids"].shape[1]
        gen = out[0][prompt_len:]
        response = self.tokenizer.decode(gen, skip_special_tokens=True).strip()

        num_tokens = len(gen)
        latency_ms = round(elapsed * 1000, 1)
        tokens_sec = round(num_tokens / elapsed, 1) if elapsed > 0 else 0.0
        curr_vram = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if self.device == "cuda" else 0.0

        return {
            "response": response,
            "prompt_tokens": prompt_len,
            "completion_tokens": num_tokens,
            "total_tokens": prompt_len + num_tokens,
            "latency_ms": latency_ms,
            "tokens_sec": tokens_sec,
            "vram_mb": curr_vram
        }


def create_handler_class(server_instance: BonsaiServer):
    """Factory creating the HTTP Request Handler wired to the server instance."""

    class DynamicBonsaiHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(HTML_TEMPLATE.encode("utf-8"))

            elif self.path in ("/api/status", "/health", "/api/health"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                data = {
                    "status": "ready",
                    "device": server_instance.device,
                    "vram_mb": server_instance.vram_allocated_mb,
                    "model": server_instance.model_identifier,
                    "endpoints": ["/v1/chat/completions", "/v1/models", "/api/chat"]
                }
                self.wfile.write(json.dumps(data).encode("utf-8"))

            elif self.path == "/v1/models":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                data = {
                    "object": "list",
                    "data": [
                        {
                            "id": "bonsai-micro-ai",
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "bonsai",
                            "root": server_instance.model_identifier
                        }
                    ]
                }
                self.wfile.write(json.dumps(data).encode("utf-8"))

            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
            try:
                data = json.loads(body)
            except Exception:
                data = {}

            # OpenAI Compatible Chat Completions Endpoint
            if self.path == "/v1/chat/completions":
                messages = data.get("messages", [])
                if not messages:
                    prompt = data.get("prompt", "")
                    messages = [{"role": "user", "content": prompt}]

                # Ensure system directive exists if not specified
                has_system = any(m.get("role") == "system" for m in messages)
                if not has_system:
                    messages = [{"role": "system", "content": server_instance.system_directive}] + messages

                max_tokens = data.get("max_tokens", 256)
                temperature = data.get("temperature", 0.0)

                gen_result = server_instance.generate(messages, max_tokens=max_tokens, temperature=temperature)

                openai_response = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": data.get("model", "bonsai-micro-ai"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": gen_result["response"]
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": gen_result["prompt_tokens"],
                        "completion_tokens": gen_result["completion_tokens"],
                        "total_tokens": gen_result["total_tokens"]
                    },
                    "bonsai_telemetry": {
                        "latency_ms": gen_result["latency_ms"],
                        "tokens_sec": gen_result["tokens_sec"],
                        "vram_mb": gen_result["vram_mb"]
                    }
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(openai_response).encode("utf-8"))

            # Legacy Web UI endpoint
            elif self.path == "/api/chat":
                prompt = data.get("prompt", "")
                messages = [
                    {"role": "system", "content": server_instance.system_directive},
                    {"role": "user", "content": prompt}
                ]
                gen_result = server_instance.generate(messages, max_tokens=256)
                res_payload = {
                    "response": gen_result["response"],
                    "latency_ms": gen_result["latency_ms"],
                    "tokens_sec": gen_result["tokens_sec"],
                    "vram_mb": gen_result["vram_mb"]
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(res_payload).encode("utf-8"))

            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass

    return DynamicBonsaiHandler


def run_server(
    target: str = "./bonsai_pruned",
    port: int = 7860,
    host: str = "127.0.0.1",
    open_browser: bool = True
):
    """Initializes and runs the Bonsai web & API server."""
    server = BonsaiServer(target_path=target, host=host, port=port)
    server.load_model()

    handler_cls = create_handler_class(server)
    httpd = ThreadedHTTPServer((host, port), handler_cls)
    server.httpd = httpd

    url = f"http://{host}:{port}"
    print(f"🌱 Bonsai Live Web UI & OpenAI API ready at: {url}")
    print(f"   • Web Interface: {url}/")
    print(f"   • OpenAI API:    {url}/v1/chat/completions")
    print(f"   • Health Status: {url}/health")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🌱 Shutting down Bonsai server cleanly...")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
