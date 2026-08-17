use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

const KERNEL_DIR: &str = "kernels";
const LFM2_SOURCE: &str = "src/model/lfm2.rs";

fn main() {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is not set"));
    let kernel_dir = manifest_dir.join(KERNEL_DIR);
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is not set"));
    let cuda_arch = env::var("LLM_CUDA_ARCH").unwrap_or_else(|_| "compute_80".to_string());

    println!("cargo:rerun-if-changed={}", kernel_dir.display());
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join(LFM2_SOURCE).display()
    );
    println!("cargo:rerun-if-env-changed=LLM_CUDA_ARCH");

    compile_cuda_dir(&kernel_dir, &out_dir, &cuda_arch);
    generate_mok_lfm2(&manifest_dir, &out_dir);
}

fn replace_once(source: &mut String, old: &str, new: &str, label: &str) {
    let occurrences = source.matches(old).count();
    assert_eq!(
        occurrences, 1,
        "expected exactly one {label} integration point, found {occurrences}"
    );
    *source = source.replacen(old, new, 1);
}

fn generate_mok_lfm2(manifest_dir: &Path, out_dir: &Path) {
    let source_path = manifest_dir.join(LFM2_SOURCE);
    let mut source = fs::read_to_string(&source_path).unwrap_or_else(|error| {
        panic!("failed to read {}: {error}", source_path.display())
    });

    const SINGLE_CALL_CALIBRATION_REFERENCE: &str = r#"                                    &positions,
                                    contiguous_prefill,
                                    None,"#;
    const SINGLE_CALL_CALIBRATION_MOK: &str = r#"                                    &positions,
                                    contiguous_prefill,
                                    next_sequence_length,
                                    None,"#;

    const SINGLE_CALL_PROFILE_REFERENCE: &str = r#"                            &positions,
                            contiguous_prefill,
                            profile.as_deref_mut(),"#;
    const SINGLE_CALL_PROFILE_MOK: &str = r#"                            &positions,
                            contiguous_prefill,
                            next_sequence_length,
                            profile.as_deref_mut(),"#;

    const SINGLE_SIGNATURE_REFERENCE: &str = r#"        positions: &Tensor<u32>,
        contiguous_prefill: bool,
        mut profile: Option<&mut ModelProfileRecorder>,"#;
    const SINGLE_SIGNATURE_MOK: &str = r#"        positions: &Tensor<u32>,
        contiguous_prefill: bool,
        context_tokens: usize,
        mut profile: Option<&mut ModelProfileRecorder>,"#;

    const SINGLE_REFERENCE: &str = r#"            || {
                query =
                    ops::rms_norm_bf16(runtime, &query, &weights.query_norm, self.config.norm_eps)?;
                key = ops::rms_norm_bf16(runtime, &key, &weights.key_norm, self.config.norm_eps)?;
                ops::rope_qk_bf16_inplace(
                    runtime,
                    &mut query,
                    &mut key,
                    &self.inv_freq,
                    positions,
                )?;
                cache.write_lfm2(runtime, &key, &value, slots)
            },"#;

    const SINGLE_MOK: &str = r#"            || {
                if contiguous_prefill {
                    query = ops::rms_norm_bf16(
                        runtime,
                        &query,
                        &weights.query_norm,
                        self.config.norm_eps,
                    )?;
                    key = ops::rms_norm_bf16(
                        runtime,
                        &key,
                        &weights.key_norm,
                        self.config.norm_eps,
                    )?;
                    ops::rope_qk_bf16_inplace(
                        runtime,
                        &mut query,
                        &mut key,
                        &self.inv_freq,
                        positions,
                    )?;
                    cache.write_lfm2(runtime, &key, &value, slots)
                } else if ops::should_use_mok_one_kernel(
                    cache.page_size().value(),
                    context_tokens,
                    1,
                ) {
                    Ok(())
                } else {
                    ops::qk_norm_rope_kv_write_decode_bf16(
                        runtime,
                        &mut query,
                        &key,
                        &value,
                        &weights.query_norm,
                        &weights.key_norm,
                        &self.inv_freq,
                        positions,
                        slots,
                        cache,
                        self.config.norm_eps,
                    )
                }
            },"#;

    const SINGLE_ATTN_REFERENCE: &str = r#"            || {
                if contiguous_prefill {
                    ops::prefill_attention_lfm2_bf16(runtime, &query, &key, &value)
                } else {
                    ops::paged_attention_lfm2_bf16(runtime, &query, cache, positions)
                }
            },"#;

    const SINGLE_ATTN_MOK: &str = r#"            || {
                if contiguous_prefill {
                    ops::prefill_attention_lfm2_bf16(runtime, &query, &key, &value)
                } else if ops::should_use_mok_one_kernel(
                    cache.page_size().value(),
                    context_tokens,
                    1,
                ) {
                    ops::fused_paged_attention_decode_lfm2_bf16(
                        runtime,
                        &query,
                        &key,
                        &value,
                        &weights.query_norm,
                        &weights.key_norm,
                        &self.inv_freq,
                        positions,
                        slots,
                        cache,
                        self.config.norm_eps,
                    )
                } else {
                    ops::paged_attention_fast_lfm2_bf16(runtime, &query, cache, positions)
                }
            },"#;

    const BATCH_REFERENCE: &str = r#"        query = ops::rms_norm_bf16(runtime, &query, &weights.query_norm, self.config.norm_eps)?;
        key = ops::rms_norm_bf16(runtime, &key, &weights.key_norm, self.config.norm_eps)?;
        ops::rope_qk_bf16_inplace(
            runtime,
            &mut query,
            &mut key,
            &self.inv_freq,
            metadata.positions(),
        )?;
        arena.write_lfm2(runtime, &key, &value, metadata.physical_slots())?;
        let attended = ops::hybrid_ragged_attention_lfm2_bf16(
            runtime,
            &query,
            &key,
            &value,
            arena,
            metadata.block_tables(),
            metadata.block_table_stride(),
            metadata.request_slots(),
            metadata.positions(),
            metadata.segment_offsets(),
        )?
        .reshape(Shape::new([num_tokens, self.config.hidden_size]))?;"#;

    const BATCH_MOK: &str = r#"        let decode_only = metadata.segment_slots().numel() == num_tokens
            && metadata.segment_offsets().numel() == num_tokens + 1;
        let one_kernel_decode = decode_only
            && ops::should_use_mok_one_kernel(
                arena.page_size().value(),
                metadata.max_context_tokens(),
                num_tokens,
            );
        let attended = if one_kernel_decode {
            ops::fused_ragged_paged_attention_decode_lfm2_bf16(
                runtime,
                &query,
                &key,
                &value,
                &weights.query_norm,
                &weights.key_norm,
                &self.inv_freq,
                metadata.block_tables(),
                metadata.block_table_stride(),
                metadata.request_slots(),
                metadata.positions(),
                metadata.physical_slots(),
                arena,
                self.config.norm_eps,
            )?
        } else if decode_only {
            ops::qk_norm_rope_kv_write_arena_decode_bf16(
                runtime,
                &mut query,
                &key,
                &value,
                &weights.query_norm,
                &weights.key_norm,
                &self.inv_freq,
                metadata.positions(),
                metadata.physical_slots(),
                arena,
                self.config.norm_eps,
            )?;
            ops::paged_ragged_attention_fast_lfm2_bf16(
                runtime,
                &query,
                arena,
                metadata.block_tables(),
                metadata.block_table_stride(),
                metadata.request_slots(),
                metadata.positions(),
            )?
        } else {
            query = ops::rms_norm_bf16(
                runtime,
                &query,
                &weights.query_norm,
                self.config.norm_eps,
            )?;
            key = ops::rms_norm_bf16(
                runtime,
                &key,
                &weights.key_norm,
                self.config.norm_eps,
            )?;
            ops::rope_qk_bf16_inplace(
                runtime,
                &mut query,
                &mut key,
                &self.inv_freq,
                metadata.positions(),
            )?;
            arena.write_lfm2(runtime, &key, &value, metadata.physical_slots())?;
            ops::hybrid_ragged_attention_lfm2_bf16(
                runtime,
                &query,
                &key,
                &value,
                arena,
                metadata.block_tables(),
                metadata.block_table_stride(),
                metadata.request_slots(),
                metadata.positions(),
                metadata.segment_offsets(),
            )?
        }
        .reshape(Shape::new([num_tokens, self.config.hidden_size]))?;"#;

    replace_once(
        &mut source,
        SINGLE_CALL_CALIBRATION_REFERENCE,
        SINGLE_CALL_CALIBRATION_MOK,
        "single-request calibration attention call",
    );
    replace_once(
        &mut source,
        SINGLE_CALL_PROFILE_REFERENCE,
        SINGLE_CALL_PROFILE_MOK,
        "single-request profiled attention call",
    );
    replace_once(
        &mut source,
        SINGLE_SIGNATURE_REFERENCE,
        SINGLE_SIGNATURE_MOK,
        "single-request attention signature",
    );
    replace_once(
        &mut source,
        SINGLE_REFERENCE,
        SINGLE_MOK,
        "single-request attention postprocess",
    );
    replace_once(
        &mut source,
        SINGLE_ATTN_REFERENCE,
        SINGLE_ATTN_MOK,
        "single-request attention dispatch",
    );
    replace_once(
        &mut source,
        BATCH_REFERENCE,
        BATCH_MOK,
        "batched attention dispatch",
    );

    let generated = out_dir.join("lfm2_mok.rs");
    fs::write(&generated, source)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", generated.display()));
}

fn compile_cuda_dir(kernel_dir: &Path, out_dir: &Path, cuda_arch: &str) {
    let entries = fs::read_dir(kernel_dir).unwrap_or_else(|err| {
        panic!(
            "failed to read kernel directory {}: {err}",
            kernel_dir.display(),
        )
    });

    for entry in entries {
        let path = entry.expect("failed to read kernel entry").path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("cu") {
            continue;
        }
        compile_ptx(&path, out_dir, cuda_arch);
    }
}

fn compile_ptx(source: &Path, out_dir: &Path, cuda_arch: &str) {
    let stem = source
        .file_stem()
        .and_then(|name| name.to_str())
        .expect("invalid CUDA filename");
    let output = out_dir.join(format!("{stem}.ptx"));

    println!("cargo:rerun-if-changed={}", source.display());

    let status = Command::new("nvcc")
        .arg("-ptx")
        .arg("-O3")
        .arg(format!("-arch={cuda_arch}"))
        .arg("-lineinfo")
        .arg(source)
        .arg("-o")
        .arg(&output)
        .status()
        .unwrap_or_else(|err| panic!("failed to execute nvcc: {err}"));

    assert!(
        status.success(),
        "nvcc failed compiling {}",
        source.display(),
    );
}
