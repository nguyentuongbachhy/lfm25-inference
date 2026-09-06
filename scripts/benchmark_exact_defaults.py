#!/usr/bin/env python3
"""
Rigorous ABBA Benchmark Suite testing the EXACT current defaults vs conservative baseline.

Configurations:
  A: Conservative Validated Runtime (speculative=0, fused_rms=false, fused_swiglu=false)
  B: Exact Current Default (speculative=3, fused_rms=true, fused_swiglu=true)
  C: Ablation SwiGLU OFF (speculative=3, fused_rms=true, fused_swiglu=false)
  D: Deterministic Fused RMS (speculative=0, fused_rms=true, fused_swiglu=false)
  E: Deterministic Both Fused (speculative=0, fused_rms=true, fused_swiglu=true)
"""

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
    "A_conservative": {
        "name": "A: Conservative Validated (spec=0, unfused)",
        "speculative_draft": 0,
        "fused_rms_fp8": False,
        "fused_swiglu_fp8": False,
    },
    "B_current_default": {
        "name": "B: EXACT Current Default (spec=3, fused_rms=true, fused_swiglu=true)",
        "speculative_draft": 3,
        "fused_rms_fp8": True,
        "fused_swiglu_fp8": True,
    },
    "C_spec3_rms_only": {
        "name": "C: Spec3 + Fused RMS Only (spec=3, fused_rms=true, fused_swiglu=false)",
        "speculative_draft": 3,
        "fused_rms_fp8": True,
        "fused_swiglu_fp8": False,
    },
    "D_spec0_rms_only": {
        "name": "D: Deterministic Fused RMS Only (spec=0, fused_rms=true, fused_swiglu=false)",
        "speculative_draft": 0,
        "fused_rms_fp8": True,
        "fused_swiglu_fp8": False,
    },
    "E_spec0_both_fused": {
        "name": "E: Deterministic Both Fused (spec=0, fused_rms=true, fused_swiglu=true)",
        "speculative_draft": 0,
        "fused_rms_fp8": True,
        "fused_swiglu_fp8": True,
    },
}

# 4-round Latin square schedule across 5 configs
SCHEDULES = [
    ["A_conservative", "B_current_default", "C_spec3_rms_only", "D_spec0_rms_only", "E_spec0_both_fused"],
    ["E_spec0_both_fused", "D_spec0_rms_only", "C_spec3_rms_only", "B_current_default", "A_conservative"],
    ["B_current_default", "D_spec0_rms_only", "A_conservative", "E_spec0_both_fused", "C_spec3_rms_only"],
    ["C_spec3_rms_only", "E_spec0_both_fused", "B_current_default", "A_conservative", "D_spec0_rms_only"],
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
        "--temperature", "0.0",
        "--speculative-draft", str(cfg["speculative_draft"]),
    ]
    if cfg["fused_rms_fp8"]:
        cmd.append("--fused-rms-fp8")
    else:
        cmd.append("--no-fused-rms-fp8")

    if cfg["fused_swiglu_fp8"]:
        cmd.append("--fused-swiglu-fp8")
    else:
        cmd.append("--no-fused-swiglu-fp8")

    cmd.append("--no-metadata-scratch")

    start_wall = time.time()
    env = dict(os.environ)
    env["LLM_CUDA_ARCH"] = "compute_120"
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
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

def percentile(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    d = k - f
    return s[f] * (1.0 - d) + s[c] * d

def main():
    print("=" * 75)
    print(" FINAL PRODUCTION RELEASE GATE: EXACT DEFAULTS ABBA BENCHMARK")
    print(f" Model: {MODEL}")
    print(f" Max new tokens: {MAX_NEW_TOKENS}")
    print(f" Configurations evaluated:")
    for k, v in CONFIGS.items():
        print(f"   [{k}]: {v['name']}")
    print("=" * 75)

    all_data = {k: [] for k in CONFIGS}
    outputs = {k: {} for k in CONFIGS}

    for p_idx, prompt in enumerate(PROMPTS):
        p_id = prompt["id"]
        print(f"\nWorkload [{p_idx + 1}/{len(PROMPTS)}] ({prompt['lang']}): '{prompt['text'][:55]}...'")

        # Warmup
        print("  Warming up GPU caches for all configurations...")
        for cfg_key in CONFIGS:
            run_trial(cfg_key, prompt["text"])

        # Order-balanced ABBA rounds
        for r_idx, schedule in enumerate(SCHEDULES):
            print(f"  Round {r_idx + 1}/{len(SCHEDULES)}: {' -> '.join([k.split('_')[0] for k in schedule])}")
            for cfg_key in schedule:
                text, metrics = run_trial(cfg_key, prompt["text"])
                all_data[cfg_key].append(metrics)
                if p_id not in outputs[cfg_key]:
                    outputs[cfg_key][p_id] = text

    print("\n" + "=" * 80)
    print(" SUMMARY METRICS ACROSS ALL 5 WORKLOADS")
    print("=" * 80)

    # Reference is A_conservative
    a_tpots = [m["tpot_mean_ms"] for m in all_data["A_conservative"]]
    base_tpot_mean = sum(a_tpots) / len(a_tpots)

    summary_rows = []
    for cfg_key, cfg in CONFIGS.items():
        data = all_data[cfg_key]
        tpots = [m["tpot_mean_ms"] for m in data]
        ttfts = [m["ttft_ms"] for m in data]
        totals = [m["total_ms"] for m in data]

        mean_tpot = sum(tpots) / len(tpots)
        p50_tpot = percentile(tpots, 50)
        p95_tpot = percentile(tpots, 95)
        mean_ttft = sum(ttfts) / len(ttfts)
        mean_total = sum(totals) / len(totals)
        speedup = base_tpot_mean / mean_tpot

        summary_rows.append({
            "key": cfg_key,
            "name": cfg["name"],
            "mean_tpot": mean_tpot,
            "p50_tpot": p50_tpot,
            "p95_tpot": p95_tpot,
            "mean_ttft": mean_ttft,
            "mean_total": mean_total,
            "speedup": speedup,
        })

    print(f"{'Configuration':<35} | {'TPOT Mean':<10} | {'P50':<8} | {'P95':<8} | {'TTFT Mean':<10} | {'Speedup':<8}")
    print("-" * 90)
    for r in summary_rows:
        print(f"{r['key']:<35} | {r['mean_tpot']:<10.3f} | {r['p50_tpot']:<8.3f} | {r['p95_tpot']:<8.3f} | {r['mean_ttft']:<10.2f} | {r['speedup']:<8.4f}x")

    print("\n" + "=" * 80)
    print(" BITWISE PARITY ANALYSIS (Golden Reference: A_conservative)")
    print("=" * 80)

    for cfg_key in CONFIGS:
        if cfg_key == "A_conservative":
            continue
        mismatches = 0
        for p_id in [p["id"] for p in PROMPTS]:
            ref = outputs["A_conservative"][p_id]
            test = outputs[cfg_key][p_id]
            if ref != test:
                mismatches += 1
        pct = ((len(PROMPTS) - mismatches) / len(PROMPTS)) * 100.0
        status = "PASS (100.00%)" if mismatches == 0 else f"FAIL ({mismatches}/{len(PROMPTS)} mismatches, {pct:.1f}%)"
        print(f"  {cfg_key:<35} vs A_conservative: {status}")

    # Also compare B vs C (effect of SwiGLU under spec=3)
    b_vs_c_mismatches = 0
    for p_id in [p["id"] for p in PROMPTS]:
        if outputs["B_current_default"][p_id] != outputs["C_spec3_rms_only"][p_id]:
            b_vs_c_mismatches += 1
    print(f"  {'B_current_default vs C_spec3_rms_only':<35} : {'PASS (100.00%)' if b_vs_c_mismatches == 0 else 'FAIL'}")

    # Compare D vs E (effect of SwiGLU under spec=0)
    d_vs_e_mismatches = 0
    for p_id in [p["id"] for p in PROMPTS]:
        if outputs["D_spec0_rms_only"][p_id] != outputs["E_spec0_both_fused"][p_id]:
            d_vs_e_mismatches += 1
    print(f"  {'D_spec0_rms_only vs E_spec0_both_fused':<35} : {'PASS (100.00%)' if d_vs_e_mismatches == 0 else 'FAIL'}")

    # Output JSON for automated reporting
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configs": CONFIGS,
        "summary": summary_rows,
    }
    with open("docs/benchmarks/exact_defaults_abba_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved full report to docs/benchmarks/exact_defaults_abba_report.json")

if __name__ == "__main__":
    main()
