use std::{ffi::c_void, mem::size_of, sync::Arc};

use anyhow::{Context as _, Result, ensure};
use cudarc::{
    cublaslt::{result, sys},
    driver::{CudaSlice, CudaStream, DevicePtr},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Fp8ScaleMode {
    Tensorwide,
    Block32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Fp8MatmulKey {
    m: usize,
    n: usize,
    k: usize,
    scale_mode: Fp8ScaleMode,
}

impl Fp8MatmulKey {
    pub(crate) fn new(m: usize, n: usize, k: usize, scale_mode: Fp8ScaleMode) -> Result<Self> {
        ensure!(m > 0, "FP8 matmul M must be > 0");
        ensure!(n > 0, "FP8 matmul N must be > 0");
        ensure!(k > 0, "FP8 matmul K must be > 0");
        ensure!(
            scale_mode != Fp8ScaleMode::Block32 || k % 32 == 0,
            "MXFP8 matmul K must be divisible by 32, got {k}",
        );

        Ok(Self {
            m,
            n,
            k,
            scale_mode,
        })
    }
}

struct MatrixLayout {
    raw: sys::cublasLtMatrixLayout_t,
}

impl MatrixLayout {
    fn row_major(data_type: sys::cudaDataType_t, rows: usize, cols: usize) -> Result<Self> {
        let rows_u64 = u64::try_from(rows).context("FP8 matrix rows exceed u64")?;
        let cols_u64 = u64::try_from(cols).context("FP8 matrix cols exceed u64")?;
        let rows_i64 = i64::try_from(rows).context("FP8 matrix rows exceed i64")?;
        let row_ld = i64::try_from(cols).context("FP8 matrix leading dimension exceeds i64")?;
        let create_ld = rows_i64.max(row_ld);

        let raw = result::create_matrix_layout(data_type, rows_u64, cols_u64, create_ld)
            .context("failed to create cuBLASLt FP8 matrix layout")?;
        let layout = Self { raw };
        let order = sys::cublasLtOrder_t::CUBLASLT_ORDER_ROW;

        unsafe {
            result::set_matrix_layout_attribute(
                layout.raw,
                sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_ORDER,
                &order as *const _ as *const c_void,
                size_of::<sys::cublasLtOrder_t>(),
            )
            .context("failed to set cuBLASLt FP8 row-major order")?;

            if create_ld != row_ld {
                result::set_matrix_layout_attribute(
                    layout.raw,
                    sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_LD,
                    &row_ld as *const _ as *const c_void,
                    size_of::<i64>(),
                )
                .context("failed to set cuBLASLt FP8 leading dimension")?;
            }
        }

        Ok(layout)
    }

    fn raw(&self) -> sys::cublasLtMatrixLayout_t {
        self.raw
    }
}

impl Drop for MatrixLayout {
    fn drop(&mut self) {
        unsafe {
            let _ = result::destroy_matrix_layout(self.raw);
        }
    }
}

struct MatmulDesc {
    raw: sys::cublasLtMatmulDesc_t,
}

impl MatmulDesc {
    fn linear_fp8(scale_mode: Fp8ScaleMode) -> Result<Self> {
        let raw = result::create_matmul_desc(
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cudaDataType_t::CUDA_R_32F,
        )
        .context("failed to create cuBLASLt FP8 matmul descriptor")?;
        let desc = Self { raw };
        let trans_a = 0_i32;
        let trans_b = 1_i32;

        unsafe {
            result::set_matmul_desc_attribute(
                desc.raw,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &trans_a as *const _ as *const c_void,
                size_of::<i32>(),
            )
            .context("failed to set cuBLASLt FP8 transA")?;
            result::set_matmul_desc_attribute(
                desc.raw,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                &trans_b as *const _ as *const c_void,
                size_of::<i32>(),
            )
            .context("failed to set cuBLASLt FP8 transB")?;

            if scale_mode == Fp8ScaleMode::Block32 {
                let mode =
                    sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;

                result::set_matmul_desc_attribute(
                    desc.raw,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                    &mode as *const _ as *const c_void,
                    size_of::<sys::cublasLtMatmulMatrixScale_t>(),
                )
                .context("failed to set cuBLASLt MXFP8 A scale mode")?;
                result::set_matmul_desc_attribute(
                    desc.raw,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                    &mode as *const _ as *const c_void,
                    size_of::<sys::cublasLtMatmulMatrixScale_t>(),
                )
                .context("failed to set cuBLASLt MXFP8 B scale mode")?;
            }
        }

        Ok(desc)
    }

    fn set_scale_pointers(&self, a: u64, b: u64) -> Result<()> {
        unsafe {
            result::set_matmul_desc_attribute(
                self.raw,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                &a as *const u64 as *const c_void,
                size_of::<u64>(),
            )
            .context("failed to set cuBLASLt FP8 A scale pointer")?;
            result::set_matmul_desc_attribute(
                self.raw,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                &b as *const u64 as *const c_void,
                size_of::<u64>(),
            )
            .context("failed to set cuBLASLt FP8 B scale pointer")?;
        }

        Ok(())
    }

    fn raw(&self) -> sys::cublasLtMatmulDesc_t {
        self.raw
    }
}

impl Drop for MatmulDesc {
    fn drop(&mut self) {
        unsafe {
            let _ = result::destroy_matmul_desc(self.raw);
        }
    }
}

struct MatmulPreference {
    raw: sys::cublasLtMatmulPreference_t,
}

impl MatmulPreference {
    fn new(workspace_size: usize) -> Result<Self> {
        let raw =
            result::create_matmul_pref().context("failed to create cuBLASLt FP8 preference")?;
        let pref = Self { raw };

        unsafe {
            result::set_matmul_pref_attribute(
                pref.raw,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &workspace_size as *const usize as *const c_void,
                size_of::<usize>(),
            )
            .context("failed to set cuBLASLt FP8 workspace preference")?;
        }

        Ok(pref)
    }
}

impl Drop for MatmulPreference {
    fn drop(&mut self) {
        unsafe {
            let _ = result::destroy_matmul_pref(self.raw);
        }
    }
}

enum ScaleStorage {
    Tensorwide {
        a: CudaSlice<f32>,
        b: CudaSlice<f32>,
    },
    Block32 {
        a: CudaSlice<u8>,
        b: CudaSlice<u8>,
    },
}

impl ScaleStorage {
    fn new(stream: &Arc<CudaStream>, key: Fp8MatmulKey) -> Result<Self> {
        match key.scale_mode {
            Fp8ScaleMode::Tensorwide => Ok(Self::Tensorwide {
                a: stream.clone_htod(&[1.0f32])?,
                b: stream.clone_htod(&[1.0f32])?,
            }),
            Fp8ScaleMode::Block32 => Ok(Self::Block32 {
                a: stream.clone_htod(&block32_unit_scales(key.m, key.k)?)?,
                b: stream.clone_htod(&block32_unit_scales(key.n, key.k)?)?,
            }),
        }
    }

    fn device_pointers(&self, stream: &Arc<CudaStream>) -> (u64, u64) {
        match self {
            Self::Tensorwide { a, b } => {
                let (a_ptr, _a_record) = a.device_ptr(stream);
                let (b_ptr, _b_record) = b.device_ptr(stream);
                (a_ptr, b_ptr)
            }
            Self::Block32 { a, b } => {
                let (a_ptr, _a_record) = a.device_ptr(stream);
                let (b_ptr, _b_record) = b.device_ptr(stream);
                (a_ptr, b_ptr)
            }
        }
    }
}

fn block32_scale_storage_len(outer: usize, inner: usize) -> Result<usize> {
    let outer_tiles = outer
        .checked_add(127)
        .context("MXFP8 outer padding overflow")?
        / 128;
    let inner_tiles = inner
        .checked_add(127)
        .context("MXFP8 inner padding overflow")?
        / 128;

    outer_tiles
        .checked_mul(inner_tiles)
        .and_then(|tiles| tiles.checked_mul(512))
        .context("MXFP8 scale storage size overflow")
}

fn block32_unit_scales(outer: usize, inner: usize) -> Result<Vec<u8>> {
    let outer_tiles = outer
        .checked_add(127)
        .context("MXFP8 outer padding overflow")?
        / 128;
    let inner_blocks = inner
        .checked_add(31)
        .context("MXFP8 inner block count overflow")?
        / 32;
    let inner_tiles = inner_blocks
        .checked_add(3)
        .context("MXFP8 inner tile count overflow")?
        / 4;
    let mut scales = vec![0_u8; block32_scale_storage_len(outer, inner)?];

    for outer_index in 0..outer {
        let outer_tile = outer_index / 128;
        let local_outer = outer_index % 128;

        for inner_block in 0..inner_blocks {
            let inner_tile = inner_block / 4;
            let local_inner = inner_block % 4;
            let tile = outer_tile
                .checked_mul(inner_tiles)
                .and_then(|value| value.checked_add(inner_tile))
                .context("MXFP8 scale tile index overflow")?;
            let local_offset = (local_outer % 32) * 16 + (local_outer / 32) * 4 + local_inner;
            let offset = tile
                .checked_mul(512)
                .and_then(|value| value.checked_add(local_offset))
                .context("MXFP8 scale offset overflow")?;

            // UE8M0 exponent bias is 127, therefore 127 encodes 2^0 = 1.
            scales[offset] = 127;
        }
    }

    ensure!(outer_tiles * inner_tiles * 512 == scales.len());

    Ok(scales)
}

pub(crate) struct Fp8MatmulPlan {
    desc: MatmulDesc,
    a_layout: MatrixLayout,
    b_layout: MatrixLayout,
    c_layout: MatrixLayout,
    algo: sys::cublasLtMatmulAlgo_t,
    _scales: ScaleStorage,
}

impl Fp8MatmulPlan {
    pub(crate) fn new(
        handle: sys::cublasLtHandle_t,
        stream: &Arc<CudaStream>,
        key: Fp8MatmulKey,
        workspace_size: usize,
    ) -> Result<Self> {
        let a_layout = MatrixLayout::row_major(sys::cudaDataType_t::CUDA_R_8F_E4M3, key.m, key.k)?;
        let b_layout = MatrixLayout::row_major(sys::cudaDataType_t::CUDA_R_8F_E4M3, key.n, key.k)?;
        let c_layout = MatrixLayout::row_major(sys::cudaDataType_t::CUDA_R_16BF, key.m, key.n)?;
        let desc = MatmulDesc::linear_fp8(key.scale_mode)?;
        let scales = ScaleStorage::new(stream, key)?;
        let (a_scale_ptr, b_scale_ptr) = scales.device_pointers(stream);
        desc.set_scale_pointers(a_scale_ptr, b_scale_ptr)?;
        let preference = MatmulPreference::new(workspace_size)?;

        let heuristic = unsafe {
            result::get_matmul_algo_heuristic(
                handle,
                desc.raw(),
                a_layout.raw(),
                b_layout.raw(),
                c_layout.raw(),
                c_layout.raw(),
                preference.raw,
            )
        }
        .context("cuBLASLt failed to select FP8 matmul algorithm")?;

        ensure!(
            heuristic.workspaceSize <= workspace_size,
            "cuBLASLt FP8 algorithm requires {} bytes, workspace has {} bytes",
            heuristic.workspaceSize,
            workspace_size,
        );

        Ok(Self {
            desc,
            a_layout,
            b_layout,
            c_layout,
            algo: heuristic.algo,
            _scales: scales,
        })
    }

    pub(crate) fn desc(&self) -> sys::cublasLtMatmulDesc_t {
        self.desc.raw()
    }

    pub(crate) fn a_layout(&self) -> sys::cublasLtMatrixLayout_t {
        self.a_layout.raw()
    }

    pub(crate) fn b_layout(&self) -> sys::cublasLtMatrixLayout_t {
        self.b_layout.raw()
    }

    pub(crate) fn c_layout(&self) -> sys::cublasLtMatrixLayout_t {
        self.c_layout.raw()
    }

    pub(crate) fn algo(&self) -> &sys::cublasLtMatmulAlgo_t {
        &self.algo
    }
}

#[cfg(test)]
mod tests {
    use super::{block32_scale_storage_len, block32_unit_scales};

    #[test]
    fn block32_scale_storage_is_tiled_and_padded() {
        assert_eq!(block32_scale_storage_len(1, 32).unwrap(), 512);
        assert_eq!(block32_scale_storage_len(128, 128).unwrap(), 512);
        assert_eq!(block32_scale_storage_len(129, 128).unwrap(), 1024);
        assert_eq!(block32_scale_storage_len(128, 129).unwrap(), 1024);
    }

    #[test]
    fn block32_scale_padding_is_zero() {
        let scales = block32_unit_scales(1, 32).unwrap();
        assert_eq!(scales[0], 127);
        assert!(scales[1..].iter().all(|value| *value == 0));

        let scales = block32_unit_scales(128, 128).unwrap();
        assert!(scales.iter().all(|value| *value == 127));
    }
}
