#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
import time

BINARY = "./target/release/lfm25-inference"
MODEL = "models/LFM2.5-1.2B-Instruct"
FP8_POLICY = "docs/benchmarks/fp8/selected-policy.json"
MAX_NEW_TOKENS = 128

PROMPTS = [
    {
        "id": "en_physics",
        "lang": "English",
        "text": "Why is the sky blue? Explain the physics behind Rayleigh scattering in detail.",
    },
    {
        "id": "vi_physics",
        "lang": "Vietnamese",
        "text": "Tại sao bầu trời có màu xanh lam? Giải thích hiện tượng tán xạ ánh sáng Rayleigh một cách chi tiết.",
    },
    {
        "id": "fr_physics",
        "lang": "French",
        "text": "Pourquoi le ciel est-il bleu? Expliquez le phénomène de diffusion de Rayleigh en détail.",
    },
    {
        "id": "de_physics",
        "lang": "German",
        "text": "Warum ist der Himmel blau? Erklären Sie die Rayleigh-Streuung im Detail.",
    },
    {
        "id": "code_rust",
        "lang": "Rust",
        "text": "Write a complete Rust function to implement binary search on a sorted slice of integers, with thorough edge case handling.",
    },
]

CONFIGS = {
    "00_baseline": {
        "name": "00: Sprint 1 Champion (No SwiGLU, No Scratch)",
        "fused_swiglu_fp8": False,
        "metadata_scratch": False,
    },
    "10_fused_swiglu": {
        "name": "10: Fused SwiGLU-FP8 Only",
        "fused_swiglu_fp8": True,
        "metadata_scratch": False,
    },
    "01_metadata_scratch": {
        "name": "01: Metadata Scratchpad Only",
        "fused_swiglu_fp8": False,
        "metadata_scratch": True,
    },
    "11_combined": {
        "name": "11: Combined Champion (SwiGLU + Scratch)",
        "fused_swiglu_fp8": True,
        "metadata_scratch": True,
    },
}

SCHEDULES = [
    ["00_baseline", "10_fused_swiglu", "01_metadata_scratch", "11_combined"],
    ["11_combined", "01_metadata_scratch", "10_fused_swiglu", "00_baseline"],
    ["01_metadata_scratch", "11_combined", "00_baseline", "10_fused_swiglu"],
    ["10_fused_swiglu", "00_baseline", "11_combined", "01_metadata_scratch"],
]

METRICS_REGEX = re.compile(
    r"prompt_tokens=(?P<prompt_tokens>\d+)\s+"
    r"completion_tokens=(?P<completion_tokens>\d+)\s+"
    r"finish_reason=(?P<finish_reason>\w+)\s+"
    r"ttft_ms=(?P<ttft_ms>[0-9.]+)\s+"
    r"tpot_mean_ms=(?P<tpot_mean_ms>[0-9.]+)\s+"
    r"total_ms=(?P<total_ms>[0-9.]+)\s+"
    r"spec_accepted=(?P<spec_accepted>\d+/\d+)"
)

def run_trial(cfg_key, prompt_text):
    cfg = CONFIGS[cfg_key]
    cmd = [
        BINARY,
        "--model", MODEL,
        "--prompt", prompt_text,
        "--max-new-tokens", str(MAX_NEW_TOKENS),
        "--fp8-policy", FP8_POLICY,
        "--page-size", "16",
        "--speculative-draft", "5",
        "--fused-rms-fp8",
    ]
    if cfg["fused_swiglu_fp8"]:
        cmd.append("--fused-swiglu-fp8")
    else:
        cmd.append("--no-fused-swiglu-fp8")

    if cfg["metadata_scratch"]:
        cmd.append("--metadata-scratch")
    else:
        cmd.append("--no-metadata-scratch")

    start_wall = time.time()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    wall_duration_s = time.time() - start_wall

    if res.returncode != 0:
        print(f"ERROR running {cfg_key}: {res.stderr}", file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

    text = res.stdout.strip()
    match = METRICS_REGEX.search(res.stderr)
    if not match:
        raise RuntimeError(f"Could not parse metrics from stderr:\n{res.stderr}")

    metrics = {
        "prompt_tokens": int(match.group("prompt_tokens")),
        "completion_tokens": int(match.group("completion_tokens")),
        "ttft_ms": float(match.group("ttft_ms")),
        "tpot_mean_ms": float(match.group("tpot_mean_ms")),
        "total_ms": float(match.group("total_ms")),
        "spec_accepted": match.group("spec_accepted"),
        "wall_duration_s": wall_duration_s,
    }
    return text, metrics

def main():
    print("==================================================================")
    print(" SPRINT 2: 2x2 FACTORIAL ABBA BENCHMARK (RTX 5060 LAPTOP GPU)")
    print(" Factors: Fused SwiGLU-FP8 x Persistent Metadata Scratchpad")
    print(" Protocol: 4-Round Latin Square (ABCD -> BCDA -> CDAB -> DABC)")
    print("==================================================================")

    all_results = {k: [] for k in CONFIGS}
    parity_failures = 0
    total_comparisons = 0

    for p_idx, prompt in enumerate(PROMPTS):
        print(f"\n--- Prompt [{p_idx + 1}/{len(PROMPTS)}] ({prompt['lang']}): {prompt['text'][:50]}... ---")

        # Warmup run for all 4 configs
        print("  Running warmup for all configs...")
        for cfg_key in CONFIGS:
            run_trial(cfg_key, prompt["text"])

        # Reference text for bitwise parity check
        ref_text = None

        # Execute 4 balanced rounds (16 trials per prompt)
        for r_idx, schedule in enumerate(SCHEDULES):
            print(f"  Round {r_idx + 1}/4 (Order: {' -> '.join(schedule)})...")
            for cfg_key in schedule:
                text, metrics = run_trial(cfg_key, prompt["text"])
                if ref_text is None:
                    ref_text = text
                else:
                    total_comparisons += 1
                    if text != ref_text:
                        print(f"  [WARNING] Bitwise mismatch detected in {cfg_key}!")
                        parity_failures += 1

                all_results[cfg_key].append({
                    "prompt_id": prompt["id"],
                    "round": r_idx + 1,
                    **metrics,
                })
                print(f"    {cfg_key:<20}: TTFT={metrics['ttft_ms']:.2f}ms, TPOT={metrics['tpot_mean_ms']:.3f}ms, Spec={metrics['spec_accepted']}")

    print("\n==================================================================")
    print(" BENCHMARK COMPLETED - AGGREGATING RESULTS")
    print("==================================================================")

    summary = {}
    baseline_tpot_mean = 0.0

    for cfg_key, trials in all_results.items():
        tpots = [t["tpot_mean_ms"] for t in trials]
        ttfts = [t["ttft_ms"] for t in trials]
        totals = [t["total_ms"] for t in trials]

        mean_tpot = sum(tpots) / len(tpots)
        sorted_tpots = sorted(tpots)
        p50_tpot = sorted_tpots[len(sorted_tpots) // 2]
        p95_tpot = sorted_tpots[int(len(sorted_tpots) * 0.95)]

        mean_ttft = sum(ttfts) / len(ttfts)
        mean_total = sum(totals) / len(totals)

        if cfg_key == "00_baseline":
            baseline_tpot_mean = mean_tpot

        summary[cfg_key] = {
            "name": CONFIGS[cfg_key]["name"],
            "trials_count": len(trials),
            "tpot_mean_ms": mean_tpot,
            "tpot_p50_ms": p50_tpot,
            "tpot_p95_ms": p95_tpot,
            "ttft_mean_ms": mean_ttft,
            "total_mean_ms": mean_total,
        }

    for cfg_key in summary:
        speedup = baseline_tpot_mean / summary[cfg_key]["tpot_mean_ms"]
        summary[cfg_key]["speedup_vs_00"] = speedup

    # Print summary table
    print(f"\n{'Configuration':<45} | {'TPOT Mean':<10} | {'TPOT P50':<10} | {'TPOT P95':<10} | {'TTFT Mean':<10} | {'Speedup':<8}")
    print("-" * 110)
    for cfg_key, s in summary.items():
        print(f"{s['name']:<45} | {s['tpot_mean_ms']:<7.3f} ms | {s['tpot_p50_ms']:<7.3f} ms | {s['tpot_p95_ms']:<7.3f} ms | {s['ttft_mean_ms']:<7.2f} ms | {s['speedup_vs_00']:<6.3f}x")

    print("\n--- Bitwise Parity Status ---")
    if parity_failures == 0:
        print(f"SUCCESS: 100.00% Bitwise Output Equivalence Across All {total_comparisons} Paired Tests!")
    else:
        print(f"FAILED: {parity_failures}/{total_comparisons} tests produced different output text!")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "NVIDIA GeForce RTX 5060 Laptop GPU (8GB GDDR6, Blackwell SM120)",
        "model": MODEL,
        "max_new_tokens": MAX_NEW_TOKENS,
        "total_prompts": len(PROMPTS),
        "trials_per_config": len(PROMPTS) * len(SCHEDULES),
        "bitwise_parity_success": parity_failures == 0,
        "total_comparisons": total_comparisons,
        "summary": summary,
        "trials": all_results,
    }

    out_path = "docs/benchmarks/abba_sprint2_report.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport written to: {out_path}")

if __name__ == "__main__":
    main()

