mod benchmark;
mod cache;
mod config;
mod cuda;
mod engine;
mod generation;
mod model;
mod ops;
mod scheduler;
mod server;
mod tensor;
mod tokenizer;
mod weights;

use std::{
    env,
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

use anyhow::{Context as _, Result, bail};

use crate::{
    cache::KvPageSize,
    engine::{Engine, EngineConfig, GenerationOptions},
    generation::SamplingConfig,
    model::DecodeProfileMode,
};

struct Args {
    model: PathBuf,
    prompt: Option<String>,
    serve: Option<String>,
    calibrate_fp8: Option<PathBuf>,
    benchmark_fp8: Option<PathBuf>,
    benchmark_batched_fp8: Option<PathBuf>,
    benchmark_serving: Option<PathBuf>,
    benchmark_hardware: Option<PathBuf>,
    benchmark_load: Option<PathBuf>,
    evaluate_fp8: Option<PathBuf>,
    benchmark_output: PathBuf,
    evaluation_output: PathBuf,
    benchmark_pairs: usize,
    fp8_eval_corpus: Option<PathBuf>,
    calibration_output: PathBuf,
    calibration_max_sequences: usize,
    calibration_max_tokens: usize,
    fp8_eval_sequences: usize,
    fp8_eval_max_tokens: usize,
    fp8_policy: Option<PathBuf>,
    max_new_tokens: usize,
    device: usize,
    page_size: KvPageSize,
    temperature: f32,
    top_k: usize,
    repetition_penalty: f32,
    seed: u64,
    decode_profile: DecodeProfileMode,
    decode_profile_warmup_steps: usize,
    decode_profile_steps: usize,
    profile_output: Option<PathBuf>,
    hardware_profile: Option<PathBuf>,
    speculative_draft: usize,
    fused_rms_fp8: bool,
    fused_swiglu_fp8: bool,
    metadata_scratch: bool,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut model = PathBuf::from("models/LFM2.5-1.2B-Instruct");
        let mut prompt = None;
        let mut serve = None;
        let mut calibrate_fp8 = None;
        let mut benchmark_fp8 = None;
        let mut benchmark_batched_fp8 = None;
        let mut benchmark_serving = None;
        let mut benchmark_hardware = None;
        let mut benchmark_load = None;
        let mut evaluate_fp8 = None;
        let mut benchmark_output = PathBuf::from("docs/fp8/e2e-benchmark.json");
        let mut evaluation_output = PathBuf::from("docs/fp8/quality-final-test.json");
        let mut benchmark_pairs = 25usize;
        let mut fp8_eval_corpus = None;
        let mut calibration_output = PathBuf::from("fp8-calibration.json");
        let mut calibration_max_sequences = 256usize;
        let mut calibration_max_tokens = 1024usize;
        let mut fp8_eval_sequences = 16usize;
        let mut fp8_eval_max_tokens = 128usize;
        let mut fp8_policy = env::var("LFM25_FP8_POLICY").ok().map(PathBuf::from);
        let mut disable_fp8_policy = false;
        let mut max_new_tokens = 64usize;
        let mut device = 0usize;
        let mut page_size = KvPageSize::P16;
        let mut temperature = 0.0f32;
        let mut top_k = 50usize;
        let mut repetition_penalty = 1.0f32;
        let mut seed = 0x4c_46_4d_32u64;
        let mut decode_profile = DecodeProfileMode::Off;
        let mut decode_profile_warmup_steps = 4usize;
        let mut decode_profile_steps = 128usize;
        let mut profile_output = None;
        let mut hardware_profile = None;
        let mut speculative_draft = env::var("LFM25_SPECULATIVE_DRAFT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3usize);
        let mut fused_rms_fp8 = env::var("LFM25_FUSED_RMS_FP8")
            .map(|v| v != "0" && v != "false")
            .unwrap_or(true);
        let mut fused_swiglu_fp8 = env::var("LFM25_FUSED_SWIGLU_FP8")
            .map(|v| v != "0" && v != "false")
            .unwrap_or(true);
        let mut metadata_scratch = env::var("LFM25_METADATA_SCRATCH")
            .map(|v| v != "0" && v != "false")
            .unwrap_or(false);
        let mut args = env::args().skip(1);

        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--model" => model = PathBuf::from(next_value(&mut args, "--model")?),
                "--prompt" => prompt = Some(next_value(&mut args, "--prompt")?),
                "--serve" => serve = Some(next_value(&mut args, "--serve")?),
                "--calibrate-fp8" => {
                    calibrate_fp8 = Some(PathBuf::from(next_value(&mut args, "--calibrate-fp8")?));
                }
                "--benchmark-fp8" => {
                    benchmark_fp8 = Some(PathBuf::from(next_value(&mut args, "--benchmark-fp8")?));
                }
                "--benchmark-batched-fp8" => {
                    benchmark_batched_fp8 = Some(PathBuf::from(next_value(
                        &mut args,
                        "--benchmark-batched-fp8",
                    )?));
                }
                "--benchmark-serving" => {
                    benchmark_serving =
                        Some(PathBuf::from(next_value(&mut args, "--benchmark-serving")?));
                }
                "--benchmark-hardware" => {
                    benchmark_hardware = Some(PathBuf::from(next_value(
                        &mut args,
                        "--benchmark-hardware",
                    )?));
                }
                "--benchmark-load" => {
                    benchmark_load =
                        Some(PathBuf::from(next_value(&mut args, "--benchmark-load")?));
                }
                "--evaluate-fp8" => {
                    evaluate_fp8 = Some(PathBuf::from(next_value(&mut args, "--evaluate-fp8")?));
                }
                "--benchmark-output" => {
                    benchmark_output = PathBuf::from(next_value(&mut args, "--benchmark-output")?);
                }
                "--evaluation-output" => {
                    evaluation_output =
                        PathBuf::from(next_value(&mut args, "--evaluation-output")?);
                }
                "--benchmark-pairs" => {
                    benchmark_pairs = next_value(&mut args, "--benchmark-pairs")?
                        .parse()
                        .context("invalid --benchmark-pairs")?;
                    if !(20..=30).contains(&benchmark_pairs) {
                        bail!("--benchmark-pairs must be in [20, 30]");
                    }
                }
                "--fp8-eval-corpus" => {
                    fp8_eval_corpus =
                        Some(PathBuf::from(next_value(&mut args, "--fp8-eval-corpus")?));
                }
                "--calibration-output" => {
                    calibration_output =
                        PathBuf::from(next_value(&mut args, "--calibration-output")?);
                }
                "--calibration-max-sequences" => {
                    calibration_max_sequences =
                        next_value(&mut args, "--calibration-max-sequences")?
                            .parse()
                            .context("invalid --calibration-max-sequences")?;
                    if calibration_max_sequences == 0 {
                        bail!("--calibration-max-sequences must be positive");
                    }
                }
                "--calibration-max-tokens" => {
                    calibration_max_tokens = next_value(&mut args, "--calibration-max-tokens")?
                        .parse()
                        .context("invalid --calibration-max-tokens")?;
                    if calibration_max_tokens == 0 {
                        bail!("--calibration-max-tokens must be positive");
                    }
                }
                "--fp8-eval-sequences" => {
                    fp8_eval_sequences = next_value(&mut args, "--fp8-eval-sequences")?
                        .parse()
                        .context("invalid --fp8-eval-sequences")?;
                    if fp8_eval_sequences == 0 {
                        bail!("--fp8-eval-sequences must be positive");
                    }
                }
                "--fp8-eval-max-tokens" => {
                    fp8_eval_max_tokens = next_value(&mut args, "--fp8-eval-max-tokens")?
                        .parse()
                        .context("invalid --fp8-eval-max-tokens")?;
                    if fp8_eval_max_tokens < 64 {
                        bail!("--fp8-eval-max-tokens must be at least 64");
                    }
                }
                "--fp8-policy" => {
                    fp8_policy = Some(PathBuf::from(next_value(&mut args, "--fp8-policy")?));
                }
                "--no-fp8-policy" | "--bf16" => {
                    disable_fp8_policy = true;
                    fp8_policy = None;
                }
                "--max-new-tokens" => {
                    max_new_tokens = next_value(&mut args, "--max-new-tokens")?
                        .parse()
                        .context("invalid --max-new-tokens")?;
                }
                "--device" => {
                    device = next_value(&mut args, "--device")?
                        .parse()
                        .context("invalid --device")?;
                }
                "--page-size" => {
                    page_size = match next_value(&mut args, "--page-size")?.as_str() {
                        "16" => KvPageSize::P16,
                        "32" => KvPageSize::P32,
                        other => bail!("--page-size must be 16 or 32, got {other}"),
                    };
                }
                "--temperature" => {
                    temperature = next_value(&mut args, "--temperature")?
                        .parse()
                        .context("invalid --temperature")?;
                }
                "--top-k" => {
                    top_k = next_value(&mut args, "--top-k")?
                        .parse()
                        .context("invalid --top-k")?;
                }
                "--repetition-penalty" => {
                    repetition_penalty = next_value(&mut args, "--repetition-penalty")?
                        .parse()
                        .context("invalid --repetition-penalty")?;
                }
                "--seed" => {
                    seed = next_value(&mut args, "--seed")?
                        .parse()
                        .context("invalid --seed")?;
                }
                "--profile-decode" => {
                    decode_profile = match next_value(&mut args, "--profile-decode")?.as_str() {
                        "coarse" => DecodeProfileMode::Coarse,
                        "detailed" => DecodeProfileMode::Detailed,
                        other => bail!("--profile-decode must be coarse or detailed, got {other}"),
                    };
                }
                "--profile-warmup-steps" => {
                    decode_profile_warmup_steps = next_value(&mut args, "--profile-warmup-steps")?
                        .parse()
                        .context("invalid --profile-warmup-steps")?;
                }
                "--profile-steps" => {
                    decode_profile_steps = next_value(&mut args, "--profile-steps")?
                        .parse()
                        .context("invalid --profile-steps")?;
                    if decode_profile_steps == 0 {
                        bail!("--profile-steps must be positive");
                    }
                }
                "--profile-output" => {
                    profile_output =
                        Some(PathBuf::from(next_value(&mut args, "--profile-output")?));
                }
                "--hardware-profile" => {
                    hardware_profile =
                        Some(PathBuf::from(next_value(&mut args, "--hardware-profile")?));
                }
                "--speculative-draft" => {
                    speculative_draft = next_value(&mut args, "--speculative-draft")?
                        .parse()
                        .context("invalid --speculative-draft")?;
                }
                "--fused-rms-fp8" => {
                    fused_rms_fp8 = true;
                }
                "--no-fused-rms-fp8" => {
                    fused_rms_fp8 = false;
                }
                "--fused-swiglu-fp8" => {
                    fused_swiglu_fp8 = true;
                }
                "--no-fused-swiglu-fp8" => {
                    fused_swiglu_fp8 = false;
                }
                "--metadata-scratch" => {
                    metadata_scratch = true;
                }
                "--no-metadata-scratch" => {
                    metadata_scratch = false;
                }
                "-h" | "--help" => {
                    println!("{}", usage());
                    std::process::exit(0);
                }
                other => bail!("unknown argument {other}\n\n{}", usage()),
            }
        }

        let modes = usize::from(prompt.is_some())
            + usize::from(serve.is_some())
            + usize::from(calibrate_fp8.is_some())
            + usize::from(benchmark_fp8.is_some())
            + usize::from(benchmark_batched_fp8.is_some())
            + usize::from(benchmark_serving.is_some())
            + usize::from(benchmark_hardware.is_some())
            + usize::from(benchmark_load.is_some())
            + usize::from(evaluate_fp8.is_some());
        if modes != 1 {
            bail!(
                "specify exactly one run mode, including --benchmark-serving\n\n{}",
                usage()
            );
        }
        if calibrate_fp8.is_some() && fp8_eval_corpus.is_none() {
            bail!("--calibrate-fp8 requires a disjoint --fp8-eval-corpus");
        }
        if calibrate_fp8.is_some() && fp8_policy.is_some() {
            bail!("calibration must use the BF16 reference path without --fp8-policy");
        }
        if benchmark_fp8.is_some() && fp8_policy.is_some() {
            bail!("--benchmark-fp8 already specifies the policy; do not also use --fp8-policy");
        }
        if benchmark_batched_fp8.is_some() && fp8_policy.is_some() {
            bail!(
                "--benchmark-batched-fp8 already specifies the policy; do not also use --fp8-policy"
            );
        }
        if evaluate_fp8.is_some() && fp8_eval_corpus.is_none() {
            bail!("--evaluate-fp8 requires --fp8-eval-corpus");
        }
        if evaluate_fp8.is_some() && fp8_policy.is_some() {
            bail!("--evaluate-fp8 already specifies the policy; do not also use --fp8-policy");
        }
        if !disable_fp8_policy
            && fp8_policy.is_none()
            && (prompt.is_some() || serve.is_some())
            && evaluate_fp8.is_none()
            && benchmark_fp8.is_none()
            && benchmark_batched_fp8.is_none()
            && calibrate_fp8.is_none()
        {
            let default_policy = PathBuf::from("docs/benchmarks/fp8/selected-policy.json");
            if default_policy.exists() {
                fp8_policy = Some(default_policy);
            }
        }
        SamplingConfig {
            temperature,
            top_k,
            repetition_penalty,
            seed,
        }
        .validate()?;
        Ok(Self {
            model,
            prompt,
            serve,
            calibrate_fp8,
            benchmark_fp8,
            benchmark_batched_fp8,
            benchmark_serving,
            benchmark_hardware,
            benchmark_load,
            evaluate_fp8,
            benchmark_output,
            evaluation_output,
            benchmark_pairs,
            fp8_eval_corpus,
            calibration_output,
            calibration_max_sequences,
            calibration_max_tokens,
            fp8_eval_sequences,
            fp8_eval_max_tokens,
            fp8_policy,
            max_new_tokens,
            device,
            page_size,
            temperature,
            top_k,
            repetition_penalty,
            seed,
            decode_profile,
            decode_profile_warmup_steps,
            decode_profile_steps,
            profile_output,
            hardware_profile,
            speculative_draft,
            fused_rms_fp8,
            fused_swiglu_fp8,
            metadata_scratch,
        })
    }
}

fn next_value(args: &mut impl Iterator<Item = String>, option: &str) -> Result<String> {
    args.next()
        .with_context(|| format!("{option} requires a value"))
}

fn write_json(path: &std::path::Path, value: &impl serde::Serialize, label: &str) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("failed to create {label} {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)
        .with_context(|| format!("failed to write {label}"))?;
    writer
        .flush()
        .with_context(|| format!("failed to flush {label}"))
}

fn usage() -> &'static str {
    "Usage:\n  llm-inference --prompt TEXT [OPTIONS]\n  llm-inference --serve ADDRESS --hardware-profile PATH [OPTIONS]\n  llm-inference --calibrate-fp8 CORPUS --fp8-eval-corpus CORPUS [OPTIONS]\n  llm-inference --evaluate-fp8 POLICY --fp8-eval-corpus CORPUS [OPTIONS]\n  llm-inference --benchmark-fp8 POLICY [OPTIONS]\n  llm-inference --benchmark-batched-fp8 POLICY [OPTIONS]\n  llm-inference --benchmark-serving OUTPUT.json [OPTIONS]\n  llm-inference --benchmark-hardware OUTPUT.json [OPTIONS]\n  llm-inference --benchmark-load OUTPUT.json --hardware-profile PATH [OPTIONS]\n\nOptions:\n  --model PATH\n  --max-new-tokens N\n  --speculative-draft N\n  --fused-rms-fp8\n  --no-fused-rms-fp8\n  --fused-swiglu-fp8\n  --no-fused-swiglu-fp8\n  --metadata-scratch\n  --no-metadata-scratch\n  --device N\n  --page-size 16|32\n  --hardware-profile PATH\n  --fp8-policy PATH\n  --no-fp8-policy / --bf16\n  --benchmark-output PATH\n  --benchmark-pairs 20..30\n  --evaluation-output PATH\n  --profile-decode coarse|detailed\n  --profile-warmup-steps N\n  --profile-steps N\n  --profile-output PATH\n  --temperature FLOAT\n  --top-k N\n  --repetition-penalty FLOAT\n  --seed N\n  --calibration-output PATH\n  --calibration-max-sequences N\n  --calibration-max-tokens N\n  --fp8-eval-corpus PATH\n  --fp8-eval-sequences N\n  --fp8-eval-max-tokens N"
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    eprintln!("loading model from {}", args.model.display());
    let mut engine = Engine::load(
        &args.model,
        args.device,
        EngineConfig {
            kv_page_size: args.page_size,
            decode_profile: args.decode_profile,
            decode_profile_warmup_steps: args.decode_profile_warmup_steps,
            decode_profile_steps: args.decode_profile_steps,
            fused_rms_fp8: args.fused_rms_fp8,
            fused_swiglu_fp8: args.fused_swiglu_fp8,
            metadata_scratch: args.metadata_scratch,
        },
    )?;
    if let Some(policy) = &args.fp8_policy {
        let enabled = engine.install_fp8_policy(policy)?;
        eprintln!(
            "installed decode-only FP8 policy {} with {} enabled sites",
            policy.display(),
            enabled
        );
    }
    if let Some(policy) = &args.benchmark_fp8 {
        let enabled = engine.install_fp8_policy(policy)?;
        eprintln!(
            "installed benchmark FP8 policy {} with {} enabled sites",
            policy.display(),
            enabled
        );
        let report = engine.benchmark_installed_fp8(
            &[16, 128, 512, 2048, 8192],
            128,
            2,
            args.benchmark_pairs,
        )?;
        write_json(
            &args.benchmark_output,
            &report,
            "interleaved FP8 benchmark report",
        )?;
        eprintln!(
            "wrote interleaved FP8 benchmark to {}",
            args.benchmark_output.display()
        );
        return Ok(());
    }
    if let Some(policy) = &args.benchmark_batched_fp8 {
        let enabled = engine.install_fp8_policy(policy)?;
        eprintln!(
            "installed batched FP8 benchmark policy {} with {} enabled sites",
            policy.display(),
            enabled
        );
        let report = engine.benchmark_batched_fp8(
            &[1, 2, 4, 8, 16, 32, 64],
            &[128, 2048],
            4,
            args.benchmark_pairs,
        )?;
        write_json(&args.benchmark_output, &report, "batched FP8 benchmark")?;
        eprintln!(
            "wrote batched FP8 benchmark to {}",
            args.benchmark_output.display()
        );
        return Ok(());
    }
    if let Some(output) = &args.benchmark_serving {
        let report = engine.benchmark_continuous_decode(&[1, 2, 4, 8, 16], &[16, 128], 4, 20)?;
        write_json(output, &report, "continuous decode benchmark")?;
        eprintln!("wrote continuous decode benchmark to {}", output.display());
        return Ok(());
    }
    if let Some(output) = &args.benchmark_hardware {
        let report = engine.benchmark_hardware_profile()?;
        write_json(output, &report, "hardware benchmark report")?;
        let cost_model_output = output.with_extension("cost-model.json");
        write_json(
            &cost_model_output,
            &report.cost_model,
            "hardware scheduler cost model",
        )?;
        eprintln!("wrote hardware benchmark to {}", output.display());
        eprintln!(
            "wrote scheduler cost model to {}",
            cost_model_output.display()
        );
        return Ok(());
    }
    if let Some(output) = &args.benchmark_load {
        let path = args
            .hardware_profile
            .as_deref()
            .context("--benchmark-load requires --hardware-profile")?;
        let cost_model = scheduler::HardwareCostModel::from_path(path)?;
        let report = benchmark::run_serving_load_benchmark(engine, cost_model)?;
        write_json(output, &report, "serving load benchmark")?;
        eprintln!("wrote serving load benchmark to {}", output.display());
        return Ok(());
    }
    if let Some(policy) = &args.evaluate_fp8 {
        let report = engine.validate_fp8_policy(
            policy,
            args.fp8_eval_corpus
                .as_deref()
                .context("missing FP8 validation corpus")?,
            args.fp8_eval_sequences,
            args.fp8_eval_max_tokens,
        )?;
        write_json(
            &args.evaluation_output,
            &report,
            "independent FP8 validation report",
        )?;
        eprintln!(
            "wrote independent FP8 validation to {}",
            args.evaluation_output.display()
        );
        return Ok(());
    }
    if let Some(corpus) = &args.calibrate_fp8 {
        let artifacts = engine.calibrate_fp8(
            corpus,
            args.fp8_eval_corpus
                .as_deref()
                .context("missing FP8 evaluation corpus")?,
            args.calibration_max_sequences,
            args.calibration_max_tokens,
            args.fp8_eval_sequences,
            args.fp8_eval_max_tokens,
        )?;
        let file = File::create(&args.calibration_output).with_context(|| {
            format!(
                "failed to create calibration report {}",
                args.calibration_output.display()
            )
        })?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &artifacts.calibration)
            .context("failed to write FP8 calibration report")?;
        writer
            .flush()
            .context("failed to flush FP8 calibration report")?;
        let outlier_path = args
            .calibration_output
            .with_file_name("calibration-outliers.json");
        write_json(
            &outlier_path,
            &artifacts.calibration.activation_outliers(40),
            "FP8 calibration outliers",
        )?;
        let gemm_error_path = args.calibration_output.with_file_name("gemm-error.json");
        let gemm_error_file = File::create(&gemm_error_path).with_context(|| {
            format!(
                "failed to create GEMM error report {}",
                gemm_error_path.display()
            )
        })?;
        let mut gemm_error_writer = BufWriter::new(gemm_error_file);
        serde_json::to_writer_pretty(&mut gemm_error_writer, &artifacts.gemm_error)
            .context("failed to write GEMM error report")?;
        gemm_error_writer
            .flush()
            .context("failed to flush GEMM error report")?;
        let policy_path = args.calibration_output.with_file_name("policies.json");
        write_json(&policy_path, &artifacts.policies, "FP8 policy report")?;
        let quality_path = args.calibration_output.with_file_name("quality.json");
        write_json(&quality_path, &artifacts.quality, "FP8 quality report")?;
        let sensitivity_path = args.calibration_output.with_file_name("sensitivity.json");
        write_json(
            &sensitivity_path,
            &artifacts.sensitivity,
            "FP8 sensitivity report",
        )?;
        let search_path = args.calibration_output.with_file_name("policy-search.json");
        write_json(
            &search_path,
            &artifacts.policy_search,
            "FP8 policy search report",
        )?;
        if let Some(selected_policy) = &artifacts.selected_policy {
            let selected_path = args
                .calibration_output
                .with_file_name("selected-policy.json");
            write_json(&selected_path, selected_policy, "selected FP8 policy")?;
            eprintln!("wrote selected FP8 policy to {}", selected_path.display());
        }
        eprintln!(
            "wrote FP8 calibration report to {}",
            args.calibration_output.display()
        );
        eprintln!(
            "wrote FP8 calibration outliers to {}",
            outlier_path.display()
        );
        eprintln!("wrote GEMM error report to {}", gemm_error_path.display());
        eprintln!("wrote FP8 policies to {}", policy_path.display());
        eprintln!("wrote FP8 quality report to {}", quality_path.display());
        eprintln!(
            "wrote FP8 sensitivity report to {}",
            sensitivity_path.display()
        );
        eprintln!("wrote FP8 policy search to {}", search_path.display());
        return Ok(());
    }
    if let Some(address) = args.serve {
        let path = args
            .hardware_profile
            .as_deref()
            .context("--serve requires --hardware-profile generated for the active GPU")?;
        let cost_model = scheduler::HardwareCostModel::from_path(path)?;
        return server::serve(engine, &address, cost_model);
    }
    let options = GenerationOptions {
        max_new_tokens: args.max_new_tokens,
        sampling: SamplingConfig {
            temperature: args.temperature,
            top_k: args.top_k,
            repetition_penalty: args.repetition_penalty,
            seed: args.seed,
        },
        speculative_draft: args.speculative_draft,
    };
    let result = engine.generate(args.prompt.as_deref().context("missing prompt")?, options)?;
    println!("{}", result.text);
    eprintln!(
        "prompt_tokens={} completion_tokens={} finish_reason={} ttft_ms={:.3} tpot_mean_ms={} total_ms={:.3} spec_accepted={}/{}",
        result.prompt_tokens,
        result.completion_tokens,
        result.finish_reason,
        result.metrics.ttft_ms,
        result
            .metrics
            .tpot_mean_ms
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "n/a".to_string()),
        result.metrics.total_ms,
        result.metrics.speculative_accepted_tokens,
        result.metrics.speculative_draft_tokens,
    );
    if let Some(profile) = &result.profile {
        eprintln!("decode_profile={}", serde_json::to_string_pretty(profile)?);
        if let Some(path) = &args.profile_output {
            write_json(path, profile, "decode profile")?;
            eprintln!("wrote decode profile to {}", path.display());
        }
    }
    Ok(())
}
