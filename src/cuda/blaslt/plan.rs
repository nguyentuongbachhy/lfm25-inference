use std::{ffi::c_void, mem::size_of};

use anyhow::{Context as _, Result, ensure};
use cudarc::cublaslt::{result, sys};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct MatmulKey {
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
}

impl MatmulKey {
    pub(crate) fn new(m: usize, n: usize, k: usize) -> Result<Self> {
        ensure!(m > 0, "matmul M must be > 0",);

        ensure!(n > 0, "matmul N must be > 0",);

        ensure!(k > 0, "matmul K must be > 0",);

        Ok(Self { m, n, k })
    }
}

struct MatrixLayout {
    raw: sys::cublasLtMatrixLayout_t,
}

impl MatrixLayout {
    fn row_major_bf16(rows: usize, cols: usize) -> Result<Self> {
        let rows_u64 = u64::try_from(rows).context("matrix rows exceed u64")?;

        let cols_u64 = u64::try_from(cols).context("matrix cols exceed u64")?;

        let rows_i64 = i64::try_from(rows).context("matrix rows exceed i64")?;

        let row_ld = i64::try_from(cols).context("matrix leading dimension exceeds i64")?;

        /*
         * MatrixLayoutCreate initially creates the descriptor
         * with the default order. Use an LD valid for both
         * column-major and our desired row-major layout first.
         */
        let create_ld = rows_i64.max(row_ld);

        let raw = result::create_matrix_layout(
            sys::cudaDataType_t::CUDA_R_16BF,
            rows_u64,
            cols_u64,
            create_ld,
        )
        .context("failed to create cuBLASLt matrix layout")?;

        let layout = Self { raw };

        let order = sys::cublasLtOrder_t::CUBLASLT_ORDER_ROW;

        unsafe {
            result::set_matrix_layout_attribute(
                layout.raw,
                sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_ORDER,
                &order as *const _ as *const c_void,
                size_of::<sys::cublasLtOrder_t>(),
            )
            .context("failed to set cuBLASLt row-major order")?;
        }

        if create_ld != row_ld {
            unsafe {
                result::set_matrix_layout_attribute(
                    layout.raw,
                    sys::cublasltMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_LD,
                    &row_ld as *const _ as *const c_void,
                    size_of::<i64>(),
                )
                .context("failed to set cuBLASLt leading dimension")?;
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
    fn linear_bf16() -> Result<Self> {
        let raw = result::create_matmul_desc(
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cudaDataType_t::CUDA_R_32F,
        )
        .context("failed to create cuBLASLt matmul descriptor")?;

        let desc = Self { raw };

        /*
         * A = X      [M, K], row-major
         * B = Weight [N, K], row-major
         *
         * Y = A * B^T
         */
        let trans_a = 0_i32;

        let trans_b = 1_i32;

        unsafe {
            result::set_matmul_desc_attribute(
                desc.raw,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &trans_a as *const _ as *const c_void,
                size_of::<i32>(),
            )
            .context("failed to set cuBLASLt transA")?;

            result::set_matmul_desc_attribute(
                desc.raw,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                &trans_b as *const _ as *const c_void,
                size_of::<i32>(),
            )
            .context("failed to set cuBLASLt transB")?;
        }

        Ok(desc)
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
        let raw = result::create_matmul_pref().context("failed to create cuBLASLt preference")?;

        let pref = Self { raw };

        unsafe {
            result::set_matmul_pref_attribute(
                pref.raw,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &workspace_size as *const usize as *const c_void,
                size_of::<usize>(),
            )
            .context("failed to set cuBLASLt workspace preference")?;
        }

        Ok(pref)
    }

    fn raw(&self) -> sys::cublasLtMatmulPreference_t {
        self.raw
    }
}

impl Drop for MatmulPreference {
    fn drop(&mut self) {
        unsafe {
            let _ = result::destroy_matmul_pref(self.raw);
        }
    }
}

pub(crate) struct MatmulPlan {
    desc: MatmulDesc,

    a_layout: MatrixLayout,
    b_layout: MatrixLayout,
    c_layout: MatrixLayout,

    algo: sys::cublasLtMatmulAlgo_t,
}

impl MatmulPlan {
    pub(crate) fn new(
        handle: sys::cublasLtHandle_t,
        key: MatmulKey,
        workspace_size: usize,
    ) -> Result<Self> {
        /*
         * Native row-major:
         *
         * A = X      [M, K]
         * B = Weight [N, K]
         * C = Y      [M, N]
         *
         * op(A) = A
         * op(B) = B^T
         *
         * [M,K] @ [K,N] -> [M,N]
         */
        let a_layout = MatrixLayout::row_major_bf16(key.m, key.k)?;

        let b_layout = MatrixLayout::row_major_bf16(key.n, key.k)?;

        let c_layout = MatrixLayout::row_major_bf16(key.m, key.n)?;

        let desc = MatmulDesc::linear_bf16()?;

        let preference = MatmulPreference::new(workspace_size)?;

        let heuristic = unsafe {
            result::get_matmul_algo_heuristic(
                handle,
                desc.raw(),
                a_layout.raw(),
                b_layout.raw(),
                c_layout.raw(),
                c_layout.raw(),
                preference.raw(),
            )
        }
        .context("cuBLASLt failed to select matmul algorithm")?;

        ensure!(
            heuristic.workspaceSize <= workspace_size,
            "cuBLASLt algorithm requires {} bytes, workspace has {} bytes",
            heuristic.workspaceSize,
            workspace_size,
        );

        Ok(Self {
            desc,
            a_layout,
            b_layout,
            c_layout,
            algo: heuristic.algo,
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
