"""
Tests for Bonsai Web Server & OpenAI-Compatible API Endpoints.
"""

import json
from unittest.mock import MagicMock, patch
from io import BytesIO
from bonsai.web_server import BonsaiServer, create_handler_class


def test_server_init(tmp_path):
    server = BonsaiServer(target_path=str(tmp_path), host="127.0.0.1", port=8000)
    assert server.port == 8000
    assert server.host == "127.0.0.1"
    assert server.model is None


def test_server_generate_mocked(tmp_path):
    server = BonsaiServer(target_path=str(tmp_path))
    server.model = MagicMock()
    server.tokenizer = MagicMock()
    server.device = "cpu"

    # Mock tokenizer apply_chat_template & return
    server.tokenizer.apply_chat_template.return_value = "System: Test\nUser: Hello\nAssistant:"
    
    mock_input_ids = MagicMock()
    mock_input_ids.shape = [1, 5]
    mock_inputs = {"input_ids": mock_input_ids}
    server.tokenizer.return_value = mock_inputs

    # Mock model generate
    mock_gen = MagicMock()
    mock_gen.__getitem__.return_value = MagicMock()
    mock_out = [MagicMock()]
    server.model.generate.return_value = mock_out

    # Mock tokenizer decode
    server.tokenizer.decode.return_value = '{"response": "cultivated answer"}'

    result = server.generate([{"role": "user", "content": "Hello"}])

    assert "response" in result
    assert "latency_ms" in result
    assert "tokens_sec" in result
    assert result["response"] == '{"response": "cultivated answer"}'


def test_handler_endpoints(tmp_path):
    server = BonsaiServer(target_path=str(tmp_path))
    server.device = "cpu"
    server.model_identifier = "test-bonsai"
    server.generate = MagicMock(return_value={
        "response": "Hello from Bonsai!",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "latency_ms": 25.0,
        "tokens_sec": 200.0,
        "vram_mb": 0.0
    })

    handler_cls = create_handler_class(server)

    # 1. Test GET /health
    req = MagicMock()
    req.makefile.return_value = BytesIO(b"")
    handler = handler_cls(req, ("127.0.0.1", 12345), MagicMock())
    handler.path = "/health"
    handler.wfile = BytesIO()
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()

    handler.do_GET()
    handler.send_response.assert_called_with(200)
    response_body = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert response_body["status"] == "ready"
    assert response_body["model"] == "test-bonsai"

    # 2. Test GET /v1/models
    handler.path = "/v1/models"
    handler.wfile = BytesIO()
    handler.do_GET()
    handler.send_response.assert_called_with(200)
    models_body = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert models_body["object"] == "list"
    assert models_body["data"][0]["id"] == "bonsai-micro-ai"

    # 3. Test POST /v1/chat/completions
    handler.path = "/v1/chat/completions"
    handler.wfile = BytesIO()
    payload = json.dumps({
        "model": "bonsai-micro-ai",
        "messages": [{"role": "user", "content": "Summarize status"}],
        "temperature": 0.1
    }).encode("utf-8")
    handler.headers = {"Content-Length": str(len(payload))}
    handler.rfile = BytesIO(payload)

    handler.do_POST()
    handler.send_response.assert_called_with(200)
    chat_resp = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert chat_resp["object"] == "chat.completion"
    assert chat_resp["choices"][0]["message"]["content"] == "Hello from Bonsai!"
    assert chat_resp["usage"]["total_tokens"] == 15
    assert "bonsai_telemetry" in chat_resp
