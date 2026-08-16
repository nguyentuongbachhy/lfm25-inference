use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

const KERNEL_DIR: &str = "kernels";

fn main() {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is not set"));

    let kernel_dir = manifest_dir.join(KERNEL_DIR);

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is not set"));

    let cuda_arch = env::var("LLM_CUDA_ARCH").unwrap_or_else(|_| "compute_80".to_string());

    println!("cargo:rerun-if-changed={}", kernel_dir.display());

    println!("cargo:rerun-if-env-changed=LLM_CUDA_ARCH");

    compile_cuda_dir(&kernel_dir, &out_dir, &cuda_arch);
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
