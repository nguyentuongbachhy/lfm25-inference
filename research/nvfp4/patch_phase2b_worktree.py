#!/usr/bin/env python3
"""Patch a detached worktree with the quality-only NVFP4 replay backend.

The committed research branch keeps production src/ untouched.  The Phase 2B
runner creates a detached worktree, invokes this patcher, and runs sampled
model propagation there.  The patch is deliberately quality-only: every
selected M=1 GEMM is replayed by an external CUTLASS SM120 executable, so its
wall time must never be interpreted as production performance.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"[nvfp4-phase2b] expected one {label}, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: patch_phase2b_worktree.py WORKTREE REPO_ROOT")
    worktree = Path(sys.argv[1]).resolve()
    root = Path(sys.argv[2]).resolve()

    cargo = worktree / "Cargo.toml"
    cargo_text = cargo.read_text()
    if "nvfp4-research" not in cargo_text:
        cargo_text = replace_once(
            cargo_text,
            "[dependencies]\n",
            "[features]\nnvfp4-research = []\n\n[dependencies]\n",
            "Cargo.toml dependencies marker",
        )
        cargo.write_text(cargo_text)

    replay_src = root / "research/nvfp4/nvfp4_replay.rs"
    replay_dst = worktree / "src/ops/nvfp4_replay.rs"
    shutil.copyfile(replay_src, replay_dst)

    ops = worktree / "src/ops/mod.rs"
    ops_text = ops.read_text()
    ops_text = replace_once(
        ops_text,
        "mod mok_dispatch;\n",
        "mod mok_dispatch;\n#[cfg(feature = \"nvfp4-research\")]\nmod nvfp4_replay;\n",
        "ops module marker",
    )
    ops_text = replace_once(
        ops_text,
        "pub(crate) use mok_dispatch::should_use_mok_one_kernel;\n",
        "pub(crate) use mok_dispatch::should_use_mok_one_kernel;\n"
        "#[cfg(feature = \"nvfp4-research\")]\n"
        "pub(crate) use nvfp4_replay::{\n"
        "    is_active_site as nvfp4_research_is_active_site, linear_nvfp4_replay,\n"
        "    set_active_sites as nvfp4_research_set_active_sites,\n"
        "};\n",
        "ops export marker",
    )
    ops.write_text(ops_text)

    model = worktree / "src/model/lfm2_base.rs"
    model_text = model.read_text()

    install_marker = '''        for site in &policy.sites {
            ensure!(
                by_site.insert(site.site.as_str(), site).is_none(),
                "duplicate FP8 policy site {}",
                site.site
            );
        }
        let mut enabled = 0usize;
'''
    install_replacement = '''        for site in &policy.sites {
            ensure!(
                by_site.insert(site.site.as_str(), site).is_none(),
                "duplicate FP8 policy site {}",
                site.site
            );
        }
        #[cfg(feature = "nvfp4-research")]
        ops::nvfp4_research_set_active_sites(
            policy
                .sites
                .iter()
                .filter(|site| site.enabled)
                .map(|site| site.site.clone())
                .collect(),
        );
        let mut enabled = 0usize;
'''
    model_text = replace_once(
        model_text, install_marker, install_replacement, "install_fp8_policy active-site hook"
    )

    gate_old = '''        let gate_up = profiled(
            runtime,
            profile.as_deref_mut(),
            ProfileRegion::MlpGateUpGemm,
            || linear_dispatch(runtime, input, &weights.feed_forward.gate_up, use_fp8),
        )?;
'''
    gate_new = '''        let gate_up = profiled(
            runtime,
            profile.as_deref_mut(),
            ProfileRegion::MlpGateUpGemm,
            || {
                #[cfg(feature = "nvfp4-research")]
                if use_fp8 {
                    let site = format!("layers.{layer}.mlp.gate_up");
                    if ops::nvfp4_research_is_active_site(&site) {
                        return ops::linear_nvfp4_replay(
                            runtime,
                            input,
                            &weights.feed_forward.gate_up.bf16,
                            &site,
                        );
                    }
                }
                linear_dispatch(runtime, input, &weights.feed_forward.gate_up, use_fp8)
            },
        )?;
'''
    model_text = replace_once(model_text, gate_old, gate_new, "MLP gate/up replay hook")

    down_old = '''        profiled(
            runtime,
            profile,
            ProfileRegion::MlpDownGemm,
            || linear_dispatch(runtime, &activated, &weights.feed_forward.down, use_fp8),
        )
'''
    down_new = '''        profiled(
            runtime,
            profile,
            ProfileRegion::MlpDownGemm,
            || {
                #[cfg(feature = "nvfp4-research")]
                if use_fp8 {
                    let site = format!("layers.{layer}.mlp.down");
                    if ops::nvfp4_research_is_active_site(&site) {
                        return ops::linear_nvfp4_replay(
                            runtime,
                            &activated,
                            &weights.feed_forward.down.bf16,
                            &site,
                        );
                    }
                }
                linear_dispatch(runtime, &activated, &weights.feed_forward.down, use_fp8)
            },
        )
'''
    model_text = replace_once(model_text, down_old, down_new, "MLP down replay hook")

    lm_old = '''        let logits = profiled(
            runtime,
            profile,
            ProfileRegion::LmHead,
            || match (use_fp8_decode, self.weights.lm_head_fp8.as_ref()) {
                (true, Some(fp8)) => ops::linear_last_row_fp8_e4m3(
                    runtime,
                    &normalized,
                    &fp8.data,
                    fp8.activation_scale,
                    fp8.weight_scale,
                ),
                _ => ops::linear_last_row_bf16(runtime, &normalized, &self.weights.embedding),
            },
        )?;
'''
    lm_new = '''        let logits = profiled(
            runtime,
            profile,
            ProfileRegion::LmHead,
            || {
                #[cfg(feature = "nvfp4-research")]
                if use_fp8_decode && ops::nvfp4_research_is_active_site("lm_head") {
                    return ops::linear_nvfp4_replay(
                        runtime,
                        &normalized,
                        &self.weights.embedding,
                        "lm_head",
                    );
                }
                match (use_fp8_decode, self.weights.lm_head_fp8.as_ref()) {
                    (true, Some(fp8)) => ops::linear_last_row_fp8_e4m3(
                        runtime,
                        &normalized,
                        &fp8.data,
                        fp8.activation_scale,
                        fp8.weight_scale,
                    ),
                    _ => ops::linear_last_row_bf16(runtime, &normalized, &self.weights.embedding),
                }
            },
        )?;
'''
    model_text = replace_once(model_text, lm_old, lm_new, "LM-head replay hook")
    model.write_text(model_text)

    runner = worktree / "src/engine/runner.rs"
    runner_text = runner.read_text()
    runner_text = runner_text.replace(
        "&& final_hidden.nrmse <= 0.12\n                && final_hidden.cosine >= 0.985;",
        "&& final_hidden.nrmse <= 0.10\n                && final_hidden.cosine >= 0.995;",
        1,
    )
    runner_text = runner_text.replace(
        'fast_gate: "relative_nll_delta<=0.5%,mean_KL<=0.05,final_hidden_nrmse<=0.12,final_hidden_cosine>=0.985,no_nonfinite",',
        'fast_gate: "nvfp4_phase2b_relative_nll_delta<=0.5%,mean_KL<=0.05,final_hidden_nrmse<=0.10,final_hidden_cosine>=0.995,no_nonfinite",',
        1,
    )

    test_module = r'''

#[cfg(all(test, feature = "nvfp4-research"))]
mod nvfp4_phase2b_research_tests {
    use super::*;
    use std::{
        env,
        fs::File,
        io::{BufReader, BufWriter, Write},
        path::PathBuf,
    };

    #[derive(Serialize)]
    struct Nvfp4Phase2bReport<'a> {
        schema_version: u32,
        backend: &'static str,
        scale_recipe: &'static str,
        quality_scope: &'static str,
        performance_scope: &'static str,
        sensitivity: &'a Fp8SensitivityReport,
        policy_search: Option<&'a Fp8PolicySearchReport>,
        policy_search_error: Option<&'a str>,
    }

    fn required_path(name: &str) -> Result<PathBuf> {
        env::var(name)
            .map(PathBuf::from)
            .with_context(|| format!("{name} is required"))
    }

    fn env_usize(name: &str, default: usize) -> Result<usize> {
        match env::var(name) {
            Ok(value) => value
                .parse::<usize>()
                .with_context(|| format!("invalid {name}")),
            Err(env::VarError::NotPresent) => Ok(default),
            Err(error) => Err(error).with_context(|| format!("failed to read {name}")),
        }
    }

    #[test]
    #[ignore = "quality-only host-GPU research work package"]
    fn nvfp4_phase2b_sampled_propagation_and_policy_search() -> Result<()> {
        let model_dir = required_path("NVFP4_PHASE2B_MODEL")?;
        let corpus_path = required_path("NVFP4_PHASE2B_CORPUS")?;
        let policy_path = required_path("NVFP4_PHASE2B_POLICY")?;
        let output_path = required_path("NVFP4_PHASE2B_OUTPUT")?;
        let requested_sequences = env_usize("NVFP4_PHASE2B_SEQUENCES", 8)?;
        let max_tokens = env_usize("NVFP4_PHASE2B_MAX_TOKENS", 128)?;
        ensure!(requested_sequences >= 4, "Phase 2B requires at least four sequences");
        ensure!(max_tokens >= 64, "Phase 2B requires at least 64 source tokens");

        let mut engine = Engine::load(&model_dir, 0, EngineConfig::default())?;
        let policy = load_fp8_policy(&policy_path)?;
        ensure!(
            !policy.sites.is_empty(),
            "NVFP4 Phase 2B policy contains no candidate sites"
        );
        let file = File::open(&corpus_path).with_context(|| {
            format!("failed to open Phase 2B corpus {}", corpus_path.display())
        })?;
        let sequences = calibration_sequences(
            BufReader::new(file),
            &engine.tokenizer,
            engine.model.config().bos_token_id,
            requested_sequences,
            max_tokens,
        )?;
        ensure!(
            sequences.len() == requested_sequences,
            "Phase 2B corpus yielded {} sequences, need {requested_sequences}",
            sequences.len()
        );

        eprintln!(
            "NVFP4 Phase 2B: {} candidate sites, {} sequences, sampled exact CUTLASS replay",
            policy.sites.len(),
            sequences.len()
        );
        let sensitivity = engine.run_fp8_sensitivity(&policy, &sequences)?;
        let (policy_search, policy_search_error) =
            match engine.search_fp8_policy(&policy, &sensitivity, &sequences) {
                Ok(search) => (Some(search), None),
                Err(error) => (None, Some(format!("{error:#}"))),
            };

        let report = Nvfp4Phase2bReport {
            schema_version: 1,
            backend: "exact_cutlass_sm120_nvfp4_external_replay",
            scale_recipe: "nearest_ue4m3_per_16",
            quality_scope: "sampled_teacher_forced_single_site_propagation_and_frontier_search",
            performance_scope: "none_external_replay_wall_time_is_invalid_for_performance",
            sensitivity: &sensitivity,
            policy_search: policy_search.as_ref(),
            policy_search_error: policy_search_error.as_deref(),
        };
        let file = File::create(&output_path).with_context(|| {
            format!("failed to create Phase 2B report {}", output_path.display())
        })?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &report)
            .context("failed to serialize Phase 2B report")?;
        writer.flush().context("failed to flush Phase 2B report")?;
        eprintln!("wrote NVFP4 Phase 2B report to {}", output_path.display());
        Ok(())
    }
}
'''
    marker = "mod nvfp4_phase2b_research_tests"
    if marker not in runner_text:
        runner_text += test_module
    runner.write_text(runner_text)

    print(f"[nvfp4-phase2b] patched detached worktree {worktree}")


if __name__ == "__main__":
    main()
