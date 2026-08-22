mod calibration;
mod evaluation;
mod fp8_analysis;
mod lfm2;
#[cfg(test)]
mod argmax_production_tests;
#[cfg(test)]
mod int8_production_tests;
#[cfg(test)]
mod int8_real_model_tests;
#[cfg(test)]
mod int8_sensitivity_tests;
#[cfg(test)]
mod w8a16_real_model_tests;
#[cfg(test)]
mod w8a16_sensitivity_tests;
mod prefix;
mod profile;
pub(crate) mod quantization;

pub use calibration::Fp8CalibrationReport;
pub(crate) use calibration::{CalibrationCollector, CalibrationPhase, CalibrationTensorKind};
pub(crate) use evaluation::{HiddenCapture, LogitMetricAccumulator, PropagationAccumulator};
pub use evaluation::{LogitDistributionMetrics, PropagationPointMetrics};
pub use fp8_analysis::Fp8GemmErrorReport;
pub(crate) use fp8_analysis::characterize_gemm_site;
pub use lfm2::Lfm2Model;
pub(crate) use lfm2::{BatchModelCache, RaggedBatchInput};
pub(crate) use prefix::ConvCheckpointPool;
pub use profile::{DecodeProfileMode, DecodeProfileReport};
pub(crate) use profile::{ModelProfileRecorder, ProfileRegion, profiled};
pub use quantization::Fp8PrecisionPolicy;
pub(crate) use quantization::PrecisionClass;