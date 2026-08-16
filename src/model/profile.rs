use std::collections::BTreeMap;

use anyhow::{Result, ensure};
use serde::Serialize;

use crate::cuda::{CudaRuntime, TimingEvent};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeProfileMode {
    Off,
    Coarse,
    Detailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ProfileRegion {
    Mlp,
    MlpGateUpGemm,
    MlpSilu,
    MlpDownGemm,
    Conv,
    ConvInProj,
    ConvKernel,
    ConvOutProj,
    Attention,
    AttnQkvProj,
    AttnPostprocess,
    AttnXqa,
    AttnOutProj,
    ResidualNorm,
    LmHead,
    Sampling,
}

impl ProfileRegion {
    fn name(self) -> &'static str {
        match self {
            Self::Mlp => "mlp_total",
            Self::MlpGateUpGemm => "mlp_gate_up_gemm",
            Self::MlpSilu => "mlp_silu",
            Self::MlpDownGemm => "mlp_down_gemm",
            Self::Conv => "conv_total",
            Self::ConvInProj => "conv_in_proj",
            Self::ConvKernel => "conv_kernel",
            Self::ConvOutProj => "conv_out_proj",
            Self::Attention => "attention_total",
            Self::AttnQkvProj => "attn_qkv_proj",
            Self::AttnPostprocess => "attn_postprocess",
            Self::AttnXqa => "attn_xqa",
            Self::AttnOutProj => "attn_out_proj",
            Self::ResidualNorm => "residual_norm",
            Self::LmHead => "lm_head",
            Self::Sampling => "sampling",
        }
    }
}

struct PendingSpan {
    region: ProfileRegion,
    start: TimingEvent,
    end: TimingEvent,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProfileComponent {
    pub name: &'static str,
    pub total_gpu_ms: f64,
    pub mean_us_per_decode_step: f64,
    pub percent_of_decode_envelope: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecodeProfileReport {
    pub mode: DecodeProfileMode,
    pub decode_steps: usize,
    pub decode_gpu_envelope_ms: f64,
    pub sum_measured_gpu_ops_ms: f64,
    pub other_cuda_ms: f64,
    pub components: Vec<ProfileComponent>,
    pub aggregates: Vec<ProfileComponent>,
}

pub(crate) struct ModelProfileRecorder {
    mode: DecodeProfileMode,
    step_start: Option<TimingEvent>,
    pending: Vec<PendingSpan>,
    totals: BTreeMap<ProfileRegion, f64>,
    envelope_ms: f64,
    decode_steps: usize,
    seen_steps: usize,
    warmup_steps: usize,
    max_profile_steps: usize,
    active_step: bool,
}

impl ModelProfileRecorder {
    pub(crate) fn new(
        mode: DecodeProfileMode,
        warmup_steps: usize,
        max_profile_steps: usize,
    ) -> Result<Option<Self>> {
        if mode == DecodeProfileMode::Off {
            return Ok(None);
        }
        ensure!(
            max_profile_steps > 0,
            "decode profile steps must be positive"
        );
        Ok(Some(Self {
            mode,
            step_start: None,
            pending: Vec::new(),
            totals: BTreeMap::new(),
            envelope_ms: 0.0,
            decode_steps: 0,
            seen_steps: 0,
            warmup_steps,
            max_profile_steps,
            active_step: false,
        }))
    }

    pub(crate) fn mode(&self) -> DecodeProfileMode {
        self.mode
    }

    pub(crate) fn start_step(&mut self, runtime: &CudaRuntime) -> Result<()> {
        ensure!(
            self.step_start.is_none(),
            "decode profile step already active"
        );
        ensure!(
            self.pending.is_empty(),
            "decode profile has unresolved spans"
        );
        self.active_step =
            self.seen_steps >= self.warmup_steps && self.decode_steps < self.max_profile_steps;
        if self.active_step {
            self.step_start = Some(runtime.record_timing_event()?);
        }
        Ok(())
    }

    pub(crate) fn region<T>(
        &mut self,
        runtime: &CudaRuntime,
        region: ProfileRegion,
        run: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        if !self.active_step {
            return run();
        }
        let start = runtime.record_timing_event()?;
        let output = run()?;
        let end = runtime.record_timing_event()?;
        self.pending.push(PendingSpan { region, start, end });
        Ok(output)
    }

    pub(crate) fn finish_step(&mut self, runtime: &CudaRuntime) -> Result<()> {
        self.seen_steps += 1;
        if !self.active_step {
            return Ok(());
        }
        let start = self
            .step_start
            .take()
            .ok_or_else(|| anyhow::anyhow!("decode profile step is not active"))?;
        let end = runtime.record_timing_event()?;
        self.envelope_ms += runtime.elapsed_ms(&start, &end)?;
        for span in self.pending.drain(..) {
            let elapsed = runtime.elapsed_ms(&span.start, &span.end)?;
            *self.totals.entry(span.region).or_default() += elapsed;
        }
        self.decode_steps += 1;
        self.active_step = false;
        Ok(())
    }

    pub(crate) fn report(self) -> Result<DecodeProfileReport> {
        ensure!(
            self.step_start.is_none(),
            "decode profile step is still active"
        );
        ensure!(
            !self.totals.is_empty(),
            "decode profile contains no measured regions"
        );
        ensure!(
            self.decode_steps > 0,
            "decode profile contains no decode steps"
        );
        let sum_measured_gpu_ops_ms = self.totals.values().sum::<f64>();
        let components = self
            .totals
            .iter()
            .map(|(&region, &total)| self.component(region.name(), total))
            .collect();
        let mut aggregate_totals = BTreeMap::<&'static str, f64>::new();
        for (&region, &total) in &self.totals {
            let aggregate = match region {
                ProfileRegion::Mlp
                | ProfileRegion::MlpGateUpGemm
                | ProfileRegion::MlpSilu
                | ProfileRegion::MlpDownGemm => "mlp_total",
                ProfileRegion::Conv
                | ProfileRegion::ConvInProj
                | ProfileRegion::ConvKernel
                | ProfileRegion::ConvOutProj => "conv_total",
                ProfileRegion::Attention
                | ProfileRegion::AttnQkvProj
                | ProfileRegion::AttnPostprocess
                | ProfileRegion::AttnXqa
                | ProfileRegion::AttnOutProj => "attention_total",
                ProfileRegion::ResidualNorm => "residual_norm",
                ProfileRegion::LmHead => "lm_head",
                ProfileRegion::Sampling => "sampling",
            };
            *aggregate_totals.entry(aggregate).or_default() += total;
        }
        let aggregates = aggregate_totals
            .into_iter()
            .map(|(name, total)| self.component(name, total))
            .collect();
        Ok(DecodeProfileReport {
            mode: self.mode,
            decode_steps: self.decode_steps,
            decode_gpu_envelope_ms: self.envelope_ms,
            sum_measured_gpu_ops_ms,
            other_cuda_ms: (self.envelope_ms - sum_measured_gpu_ops_ms).max(0.0),
            components,
            aggregates,
        })
    }

    pub(crate) fn has_steps(&self) -> bool {
        self.decode_steps > 0
    }

    fn component(&self, name: &'static str, total_gpu_ms: f64) -> ProfileComponent {
        ProfileComponent {
            name,
            total_gpu_ms,
            mean_us_per_decode_step: total_gpu_ms * 1000.0 / self.decode_steps as f64,
            percent_of_decode_envelope: if self.envelope_ms > 0.0 {
                total_gpu_ms * 100.0 / self.envelope_ms
            } else {
                0.0
            },
        }
    }
}

pub(crate) fn profiled<T>(
    runtime: &CudaRuntime,
    profile: Option<&mut ModelProfileRecorder>,
    region: ProfileRegion,
    run: impl FnOnce() -> Result<T>,
) -> Result<T> {
    match profile {
        Some(profile) => profile.region(runtime, region, run),
        None => run(),
    }
}
