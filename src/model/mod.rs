mod calibration;
mod evaluation;
mod fp8_analysis;
mod lfm2 {
    include!(concat!(env!("OUT_DIR"), "/lfm2_mok.rs"));
}
mod profile;
pub(crate) mod quantization;

pub use calibration::Fp8CalibrationReport;
pub(crate) use calibration::{CalibrationCollector, CalibrationPhase, CalibrationTensorKind};
pub(crate) use evaluation::{HiddenCapture, LogitMetricAccumulator, PropagationAccumulator};
pub use evaluation::{LogitDistributionMetrics, PropagationPointMetrics};
pub use fp8_analysis::Fp8GemmErrorReport;
pub(crate) use fp8_analysis::characterize_gemm_site;
pub(crate) use lfm2::BatchModelCache;
pub use lfm2::Lfm2Model;
#[allow(unused_imports)]
pub use profile::{DecodeProfileMode, DecodeProfileReport, ProfileComponent};
pub(crate) use profile::{ModelProfileRecorder, ProfileRegion, profiled};
pub use quantization::Fp8PrecisionPolicy;
pub(crate) use quantization::PrecisionClass;
