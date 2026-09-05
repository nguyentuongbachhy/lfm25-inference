#!/usr/bin/env python3
"""
Continuous Batching Concurrency Scaling Benchmark.
Measures aggregate tokens/sec and TTFT across concurrent client streams (C = 1, 2, 4, 8).
"""

import sys
import os
import json
import urllib.request
import time
from concurrent.futures import ThreadPoolExecutor

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else os.getenv("SERVER_URL", "http://127.0.0.1:8088")
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

PROMPTS = [
    "Viết 3 câu về lợi ích của trí tuệ nhân tạo.",
    "Kể tên 5 thành phố lớn ở châu Á và điểm nổi bật.",
    "Giải thích khái niệm máy học một cách đơn giản nhất.",
    "Tại sao nước biển lại có vị mặn? Giải thích ngắn gọn.",
    "Nêu 3 thói quen tốt để cải thiện sức khỏe mỗi ngày.",
    "Mặt trăng quay quanh Trái đất mất bao lâu và ảnh hưởng gì?",
    "Tại sao lá cây lại có màu xanh vào mùa xuân và hè?",
    "Nêu 4 nguyên tắc cơ bản trong lập trình sạch (Clean Code)."
]

def run_single_request(prompt, max_tokens=50):
    payload = json.dumps({
        "model": "LFM2.5-1.2B-Instruct",
        "messages": [
            {"role": "system", "content": "Bạn là trợ lý AI hữu ích. Trả lời súc tích."},
            {"role": "user", "content": prompt}
        ],
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
    tokens = 0
    with opener.open(req) as resp:
        for line in resp:
            line_str = line.decode().strip()
            if line_str.startswith("data: ") and not line_str.startswith("data: [DONE]"):
                data = json.loads(line_str[6:])
                delta = data["choices"][0]["delta"].get("content", "")
                if delta:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    tokens += 1
    t1 = time.perf_counter()
    ttft_ms = (first_token_time - t0) * 1000.0 if first_token_time else 0.0
    total_sec = t1 - t0
    return {
        "tokens": tokens,
        "ttft_ms": ttft_ms,
        "total_sec": total_sec,
    }

def bench_concurrency(concurrency, max_tokens=50):
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(concurrency)]
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(lambda p: run_single_request(p, max_tokens), prompts))
    t_end = time.perf_counter()
    wall_clock = t_end - t_start

    total_tokens = sum(r["tokens"] for r in results)
    mean_ttft = sum(r["ttft_ms"] for r in results) / len(results)
    aggregate_tps = total_tokens / wall_clock if wall_clock > 0 else 0.0
    mean_stream_tps = aggregate_tps / concurrency

    return {
        "concurrency": concurrency,
        "wall_clock": wall_clock,
        "total_tokens": total_tokens,
        "aggregate_tps": aggregate_tps,
        "mean_stream_tps": mean_stream_tps,
        "mean_ttft_ms": mean_ttft,
    }

def main():
    print("=" * 72)
    print(" CONTINUOUS BATCHING CONCURRENCY SCALING BENCHMARK")
    print(f" Target: {BASE_URL}")
    print("=" * 72)

    print("Warmup request (C=1)...")
    run_single_request("Xin chào", max_tokens=10)

    summary = []
    for c in [1, 2, 4, 8]:
        print(f"Running Concurrency C = {c} ...", end="", flush=True)
        res = bench_concurrency(c, max_tokens=50)
        summary.append(res)
        print(f" done! (Agg: {res['aggregate_tps']:.1f} tok/s, TTFT: {res['mean_ttft_ms']:.1f} ms)")

    print("\n" + "=" * 72)
    print(f"{'Concurrency':<12} | {'Tokens':<8} | {'Wall (s)':<10} | {'Agg TPS':<14} | {'Mean TTFT (ms)':<15}")
    print("-" * 72)
    for s in summary:
        print(f"{s['concurrency']:<12} | {s['total_tokens']:<8} | {s['wall_clock']:<10.2f} | {s['aggregate_tps']:<14.1f} | {s['mean_ttft_ms']:<15.2f}")
    print("=" * 72)

if __name__ == "__main__":
    main()
