/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! End-to-end coverage for `rustc_codegen_llvm::mlir::target::resolve`:
//! `--target`'s architecture selects the Triton backend (Cuda/Riscv), and
//! `-C target-cpu` is validated per backend instead of being silently
//! reinterpreted (see teenyc-j3a).

#![feature(rustc_private)]

use std::env;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use rustc_driver::{Callbacks, run_compiler};
use rustc_interface::interface;

struct MlirBackendCallbacks;

impl Callbacks for MlirBackendCallbacks {
    fn config(&mut self, config: &mut interface::Config) {
        config.make_codegen_backend = Some(Box::new(|_sess: &rustc_session::Session| {
            rustc_codegen_llvm::mlir::MlirCodegenBackend::new()
        }));
    }
}

/// Compiles `filename` for `target`, with `extra_args` appended (e.g.
/// `-C target-cpu=...`). Returns `Err` if compilation panicked (which is how
/// `sess.dcx().fatal(..)` surfaces through `run_compiler` when used as a
/// library rather than through the `rustc` binary's own exit-code wrapper).
fn try_compile(filename: &Path, target: &str, output_name: &str, extra_args: &[&str]) -> Result<(), String> {
    let output_path = PathBuf::from("/tmp").join(format!("kernel-{output_name}.asm"));

    unsafe {
        env::set_var("CFG_VERSION", "tg-1.90.0");
    }

    let mut args = vec![
        "/home/arshadm/.cargo/bin/rustc".to_string(),
        filename.display().to_string(),
        "-Copt-level=3".to_string(),
        "-Cpanic=abort".to_string(),
        format!("-o{}", output_path.display()),
        format!("--target={target}"),
        "--crate-type=lib".to_string(),
        "-C".to_string(),
        "overflow-checks=off".to_string(),
        "--frontend=triton".to_string(),
    ];
    args.extend(extra_args.iter().map(|s| s.to_string()));

    let mut callbacks = MlirBackendCallbacks;
    panic::catch_unwind(AssertUnwindSafe(|| run_compiler(&args, &mut callbacks)))
        .map_err(|_| "compilation panicked".to_string())
}

fn data_file(name: &str) -> PathBuf {
    env::current_dir().unwrap().join("tests/data").join(name)
}

#[test]
fn cuda_valid_target_cpu_succeeds() {
    let src = data_file("triton_relu.rs");
    let result = try_compile(&src, "nvptx64-nvidia-cuda", "cuda_valid_cpu", &[
        "-C",
        "target-cpu=sm_90",
    ]);
    assert!(result.is_ok(), "expected sm_90 to be accepted: {result:?}");
}

#[test]
fn cuda_invalid_target_cpu_is_rejected() {
    // Before teenyc-j3a, an unrecognized -C target-cpu silently defaulted to
    // capability 90 instead of being rejected. A RISC-V-shaped cpu string is
    // the sharpest case: it must never be misread as a CUDA capability.
    let src = data_file("triton_relu.rs");
    let result = try_compile(&src, "nvptx64-nvidia-cuda", "cuda_invalid_cpu", &[
        "-C",
        "target-cpu=generic-rvv1.0",
    ]);
    assert!(result.is_err(), "expected an invalid CUDA target-cpu to be rejected");
}

#[test]
fn riscv_target_compiles_placeholder_kernel_end_to_end() {
    // riscv64-generic selects TargetBackend::Riscv (see
    // rustc_target::spec::targets::riscv64_generic and
    // rustc_codegen_llvm::mlir::target::resolve). RiscvBackend doesn't lower
    // the incoming module yet (makeTTIR/makeTTGIR/makeLLIR are no-ops) --
    // makeLLVMIR instead synthesizes a placeholder `void @<name>()` kernel,
    // which makeASM/makeBIN then compile for real through LLVM's RISC-V
    // backend and link the result into a shared library via ld.lld.
    // compile_module retrieves those bytes via get_bin_bytes() and
    // write_compiled_module prefers them over the ASM text, so the output
    // file below is the actual linked .so, not just assembly -- this was
    // manually verified further (outside this test) by cross-compiling a
    // dlopen(3)/dlsym(3) harness for riscv64-linux-gnu and running it under
    // `qemu-riscv64`: it loads this exact .so and successfully calls the
    // exported `riscv_kernel` symbol.
    let src = data_file("triton_relu.rs");
    let result = try_compile(&src, "riscv64-generic", "riscv_stub", &[]);
    assert!(result.is_ok(), "expected the RISC-V placeholder pipeline to succeed: {result:?}");

    let bytes = std::fs::read("/tmp/kernel-riscv_stub.asm").expect("read compiled output");
    assert_eq!(&bytes[..4], b"\x7fELF", "expected a real ELF file, not assembly text");
    // e_type at offset 16 (u16 LE): ET_DYN (3) for a shared object.
    assert_eq!(u16::from_le_bytes([bytes[16], bytes[17]]), 3, "expected ET_DYN (shared object)");
    // e_machine at offset 18 (u16 LE): EM_RISCV (243).
    assert_eq!(u16::from_le_bytes([bytes[18], bytes[19]]), 243, "expected EM_RISCV");
}
