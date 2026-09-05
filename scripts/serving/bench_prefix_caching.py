#!/usr/bin/env python3
"""
Multi-Turn Radix Tree Prefix Caching Benchmark.
Measures TTFT on Turn 1 (Cold Prefill) vs Turn 2 (Cached Prefix Hit).
"""

import sys
import os
import json
import urllib.request
import time

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else os.getenv("SERVER_URL", "http://127.0.0.1:8088")
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def send_chat_stream(messages, max_tokens=30):
    payload = json.dumps({
        "model": "LFM2.5-1.2B-Instruct",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    first_token_time = None
    chunks = 0
    full_text = ""
    with opener.open(req) as resp:
        for line in resp:
            line_str = line.decode().strip()
            if line_str.startswith("data: ") and not line_str.startswith("data: [DONE]"):
                data = json.loads(line_str[6:])
                delta = data["choices"][0]["delta"].get("content", "")
                if delta:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    chunks += 1
                    full_text += delta
    t1 = time.perf_counter()
    ttft_ms = (first_token_time - t0) * 1000.0 if first_token_time else 0.0
    total_ms = (t1 - t0) * 1000.0
    return {
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "tokens": chunks,
        "text": full_text
    }

def main():
    print("=" * 65)
    print(" MULTI-TURN RADIX TREE PREFIX CACHING BENCHMARK")
    print(f" Target: {BASE_URL}")
    print("=" * 65)
    
    # ~500-600 token context
    context_doc = (
        "Liquid AI builds state-of-the-art non-transformer and hybrid foundation models. "
        "The LFM 2.5 architecture combines gated linear convolution (short convolution) with "
        "multi-order gated state-space layers and grouped-query attention. "
        "The 1.2B model features 24 layers, 16 attention heads with 2 KV heads, "
        "and a hidden dimension of 2048. It is trained on diverse multilingual datasets, "
        "demonstrating exceptional speed and quality for edge and server workloads. "
    ) * 10

    messages_turn1 = [
        {"role": "system", "content": "You are a helpful AI assistant. Answer based on the provided text."},
        {"role": "user", "content": f"Document: {context_doc}\n\nQuestion 1: What is the hidden dimension of the LFM 2.5 1.2B model?"}
    ]

    print("\n[Turn 1] Sending prompt with ~600 tokens context (Cold Prefill)...")
    res1 = send_chat_stream(messages_turn1, max_tokens=25)
    print(f"  ✓ TTFT: {res1['ttft_ms']:.2f} ms")
    print(f"  ✓ Total time: {res1['total_ms']:.2f} ms")
    print(f"  ✓ Generated: {res1['text'].strip()}")

    messages_turn2 = [
        {"role": "system", "content": "You are a helpful AI assistant. Answer based on the provided text."},
        {"role": "user", "content": f"Document: {context_doc}\n\nQuestion 1: What is the hidden dimension of the LFM 2.5 1.2B model?"},
        {"role": "assistant", "content": res1['text'].strip()},
        {"role": "user", "content": "Question 2: How many layers does it have?"}
    ]

    print("\n[Turn 2] Sending follow-up with cached context (Radix Tree Prefix Cache Hit)...")
    res2 = send_chat_stream(messages_turn2, max_tokens=25)
    print(f"  ✓ TTFT: {res2['ttft_ms']:.2f} ms")
    print(f"  ✓ Total time: {res2['total_ms']:.2f} ms")
    print(f"  ✓ Generated: {res2['text'].strip()}")

    speedup = res1['ttft_ms'] / max(res2['ttft_ms'], 0.001)
    print("\n" + "-" * 65)
    print(f" Prefix Caching Speedup: {speedup:.2f}x (Cold TTFT: {res1['ttft_ms']:.2f} ms -> Hit TTFT: {res2['ttft_ms']:.2f} ms)")
    print("-" * 65)

if __name__ == "__main__":
    main()
