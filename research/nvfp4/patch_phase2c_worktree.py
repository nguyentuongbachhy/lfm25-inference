#!/usr/bin/env python3
"""Patch a detached worktree for Phase 2C sampled hybrid confirmation.

Phase 2C reuses the Phase 2B exact CUTLASS replay backend, but installs the
validated production E4M3 policy and intercepts only an explicit NVFP4
allowlist. This lets the same sampled evaluator compare E4M3 against
E4M3+NVFP4 replacements without committing production runtime changes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: patch_phase2c_worktree.py WORKTREE REPO_ROOT")
    worktree = Path(sys.argv[1]).resolve()
    root = Path(sys.argv[2]).resolve()

    subprocess.run(
        [
            sys.executable,
            str(root / "research/nvfp4/patch_phase2b_worktree.py"),
            str(worktree),
            str(root),
        ],
        check=True,
    )

    runner = worktree / "src/engine/runner.rs"
    text = runner.read_text()
    marker = "mod nvfp4_phase2c_research_tests"
    if marker in text:
        print(f"[nvfp4-phase2c] detached worktree already patched: {worktree}")
        return

    module = r'''

#[cfg(all(test, feature = "nvfp4-research"))]
mod nvfp4_phase2c_research_tests {
    use super::*;
    use std::{
        env,
        fs::File,
        io::{BufReader, BufWriter, Write},
        path::PathBuf,
    };

    #[derive(Serialize)]
    struct Nvfp4Phase2cReport<'a> {
        schema_version: u32,
        backend: &'static str,
        quality_scope: &'static str,
        performance_scope: &'static str,
        replay_allowlist: String,
        evaluation_sequences: usize,
        positions_per_sequence: usize,
        quality: &'a Fp8PolicyQualityReport,
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
    #[ignore = "quality-only host-GPU Phase 2C confirmation"]
    fn nvfp4_phase2c_sampled_hybrid_quality() -> Result<()> {
        let model_dir = required_path("NVFP4_PHASE2C_MODEL")?;
        let corpus_path = required_path("NVFP4_PHASE2C_CORPUS")?;
        let policy_path = required_path("NVFP4_PHASE2C_POLICY")?;
        let output_path = required_path("NVFP4_PHASE2C_OUTPUT")?;
        let requested_sequences = env_usize("NVFP4_PHASE2C_SEQUENCES", 8)?;
        let max_tokens = env_usize("NVFP4_PHASE2C_MAX_TOKENS", 256)?;
        let positions_per_sequence = env_usize("NVFP4_PHASE2C_POSITIONS", 8)?;
        ensure!(requested_sequences >= 8, "Phase 2C requires at least eight sequences");
        ensure!(max_tokens >= 128, "Phase 2C requires at least 128 source tokens");
        ensure!(positions_per_sequence >= 8, "Phase 2C requires at least eight sampled positions");

        let mut engine = Engine::load(&model_dir, 0, EngineConfig::default())?;
        let policy = load_fp8_policy(&policy_path)?;
        let enabled = policy.sites.iter().filter(|site| site.enabled).count();
        ensure!(enabled > 0, "Phase 2C production E4M3 policy enables no sites");

        let file = File::open(&corpus_path).with_context(|| {
            format!("failed to open Phase 2C corpus {}", corpus_path.display())
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
            "Phase 2C corpus yielded {} sequences, need {requested_sequences}",
            sequences.len()
        );

        let allowlist = env::var("NVFP4_REPLAY_ALLOWLIST").unwrap_or_default();
        eprintln!(
            "NVFP4 Phase 2C: production_policy_sites={} replay_allowlist='{}' sequences={} positions={}",
            enabled,
            allowlist,
            sequences.len(),
            positions_per_sequence,
        );
        let quality =
            engine.evaluate_fp8_policy_sampled(&policy, &sequences, positions_per_sequence)?;
        let report = Nvfp4Phase2cReport {
            schema_version: 1,
            backend: "production_e4m3_plus_exact_cutlass_sm120_nvfp4_external_replay",
            quality_scope: "sampled_teacher_forced_disjoint_test_confirmation",
            performance_scope: "none_external_replay_wall_time_is_invalid_for_performance",
            replay_allowlist: allowlist,
            evaluation_sequences: sequences.len(),
            positions_per_sequence,
            quality: &quality,
        };
        let file = File::create(&output_path).with_context(|| {
            format!("failed to create Phase 2C report {}", output_path.display())
        })?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &report)
            .context("failed to serialize Phase 2C report")?;
        writer.flush().context("failed to flush Phase 2C report")?;
        eprintln!("wrote NVFP4 Phase 2C report to {}", output_path.display());
        Ok(())
    }
}
'''
    runner.write_text(text + module)
    print(f"[nvfp4-phase2c] patched detached worktree {worktree}")


if __name__ == "__main__":
    main()
