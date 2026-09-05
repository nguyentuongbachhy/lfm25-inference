#!/usr/bin/env python3
"""
Comprehensive test suite verifying all OpenAI and Ollama API endpoints for lfm25-inference.
Supports custom base URL via argument or environment variable (default: http://127.0.0.1:8088).
"""

import sys
import os
import json
import urllib.request
import urllib.error
import time

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else os.getenv("SERVER_URL", "http://127.0.0.1:8088")

# Force direct connection without system HTTP proxy
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def test_health_and_version():
    print("--> [Generic] Testing GET /health and /version ...")
    with opener.open(f"{BASE_URL}/health") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data.get("status") == "ok"
        print("  ✓ /health OK:", data)

    with opener.open(f"{BASE_URL}/version") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        print("  ✓ /version OK:", data)

# ----------------- OpenAI Suite -----------------

def test_openai_models():
    print("--> [OpenAI] Testing GET /v1/models ...")
    with opener.open(f"{BASE_URL}/v1/models") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data.get("object") == "list"
        models = [m["id"] for m in data.get("data", [])]
        print("  ✓ /v1/models returned:", models)
        assert len(models) > 0

def test_openai_chat_completion():
    print("--> [OpenAI] Testing POST /v1/chat/completions (non-streaming) ...")
    payload = json.dumps({
        "model": "LFM2.5-1.2B-Instruct",
        "messages": [
            {"role": "user", "content": "1 + 1 bằng mấy? Trả lời thật ngắn gọn."}
        ],
        "max_tokens": 20,
        "temperature": 0.0
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with opener.open(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        content = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        print(f"  ✓ Response: '{content}'")
        print(f"  ✓ Usage: prompt={usage.get('prompt_tokens')}, completion={usage.get('completion_tokens')}")
        assert len(content) > 0

def test_openai_chat_streaming():
    print("--> [OpenAI] Testing POST /v1/chat/completions (streaming SSE) ...")
    payload = json.dumps({
        "model": "LFM2.5-1.2B-Instruct",
        "messages": [
            {"role": "user", "content": "Đếm từ 1 đến 3."}
        ],
        "max_tokens": 20,
        "temperature": 0.0,
        "stream": True
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with opener.open(req) as resp:
        assert resp.status == 200
        assert "text/event-stream" in resp.headers.get("Content-Type", "")
        received_chunks = 0
        streamed_text = ""
        done = False
        for line in resp:
            line_str = line.decode().strip()
            if line_str == "data: [DONE]":
                done = True
                continue
            if line_str.startswith("data: "):
                data = json.loads(line_str[6:])
                delta = data["choices"][0]["delta"].get("content", "")
                if delta:
                    received_chunks += 1
                    streamed_text += delta
        print(f"  ✓ Streamed {received_chunks} chunks: '{streamed_text.strip()}'")
        assert done, "Did not receive [DONE] marker"

def test_openai_completions():
    print("--> [OpenAI] Testing POST /v1/completions (raw completion) ...")
    payload = json.dumps({
        "prompt": "Liquid AI creates",
        "max_tokens": 15,
        "temperature": 0.0
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with opener.open(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        text = data["choices"][0]["text"].strip()
        print(f"  ✓ Text completion: '{text}'")
        assert len(text) > 0

def test_openai_structured_json():
    print("--> [OpenAI] Testing Structured Output (JSON Schema) ...")
    payload = json.dumps({
        "model": "LFM2.5-1.2B-Instruct",
        "messages": [
            {"role": "user", "content": "Thủ đô của Việt Nam là gì và ở khu vực nào?"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "capital_info",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "capital": {"type": "string"},
                        "region": {"type": "string"}
                    },
                    "required": ["capital", "region"]
                }
            }
        },
        "max_tokens": 64,
        "temperature": 0.0
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with opener.open(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        raw_content = data["choices"][0]["message"]["content"].strip()
        print("  ✓ Raw output:", raw_content)
        parsed = json.loads(raw_content)
        if "properties" in parsed and "capital" not in parsed:
            parsed = parsed["properties"]
        assert "capital" in parsed
        print(f"  ✓ Valid JSON parsed: capital='{parsed.get('capital')}', region='{parsed.get('region')}'")

# ----------------- Ollama Suite -----------------

def test_ollama_tags():
    print("--> [Ollama] Testing GET /api/tags ...")
    with opener.open(f"{BASE_URL}/api/tags") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        models = [m["name"] for m in data.get("models", [])]
        print("  ✓ Installed Ollama models:", models)
        assert len(models) > 0

def test_ollama_version():
    print("--> [Ollama] Testing GET /api/version ...")
    with opener.open(f"{BASE_URL}/api/version") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        print("  ✓ Ollama version:", data.get("version"))
        assert "version" in data

def test_ollama_show():
    print("--> [Ollama] Testing POST /api/show ...")
    payload = json.dumps({"name": "LFM2.5-1.2B-Instruct"}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/api/show",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with opener.open(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        info = data.get("model_info", {})
        print(f"  ✓ Architecture: {info.get('general.architecture')}, Context: {info.get('context_length')}")
        assert info.get("context_length") == 32768

def test_ollama_chat_streaming():
    print("--> [Ollama] Testing POST /api/chat (NDJSON Streaming) ...")
    payload = json.dumps({
        "model": "LFM2.5-1.2B-Instruct",
        "messages": [
            {"role": "user", "content": "Nói 'Xin chào Việt Nam'"}
        ],
        "options": {"num_predict": 25, "temperature": 0.0}
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with opener.open(req) as resp:
        assert resp.status == 200
        assert "application/x-ndjson" in resp.headers.get("Content-Type", "")
        streamed_text = ""
        final_chunk = None
        for line in resp:
            line_str = line.decode().strip()
            if not line_str:
                continue
            item = json.loads(line_str)
            if not item.get("done", False):
                streamed_text += item.get("message", {}).get("content", "")
            else:
                final_chunk = item
        print(f"  ✓ Streamed NDJSON text: '{streamed_text.strip()}'")
        assert final_chunk is not None and final_chunk.get("done") is True
        print(f"  ✓ Stats: eval_tokens={final_chunk.get('eval_count')}, total_duration={final_chunk.get('total_duration')} ns")

def test_ollama_chat_non_streaming():
    print("--> [Ollama] Testing POST /api/chat (non-streaming) ...")
    payload = json.dumps({
        "model": "LFM2.5-1.2B-Instruct",
        "stream": False,
        "messages": [
            {"role": "user", "content": "Mặt trời mọc ở hướng nào?"}
        ],
        "options": {"num_predict": 20, "temperature": 0.0}
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with opener.open(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        content = data.get("message", {}).get("content", "").strip()
        print(f"  ✓ Non-streaming answer: '{content}'")
        assert data.get("done") is True

def test_ollama_generate():
    print("--> [Ollama] Testing POST /api/generate ...")
    payload = json.dumps({
        "model": "LFM2.5-1.2B-Instruct",
        "prompt": "Artificial intelligence is",
        "stream": False,
        "options": {"num_predict": 15, "temperature": 0.0}
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with opener.open(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        resp_text = data.get("response", "").strip()
        print(f"  ✓ Generate response: '{resp_text}'")
        assert len(resp_text) > 0

def main():
    print_header(f"TESTING ALL ENDPOINTS ON {BASE_URL}")
    test_health_and_version()
    
    print_header("OPENAI ECOSYSTEM TESTS")
    test_openai_models()
    test_openai_chat_completion()
    test_openai_chat_streaming()
    test_openai_completions()
    test_openai_structured_json()

    print_header("OLLAMA ECOSYSTEM TESTS")
    test_ollama_tags()
    test_ollama_version()
    test_ollama_show()
    test_ollama_chat_streaming()
    test_ollama_chat_non_streaming()
    test_ollama_generate()

    print("\n" + "=" * 60)
    print(" 🎉 ALL OPENAI AND OLLAMA ENDPOINTS VERIFIED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()

