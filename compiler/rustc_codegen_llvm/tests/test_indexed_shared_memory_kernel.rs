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

//! teenygrad-3w0.10: compile `tests/data/triton_indexed_shared_memory.rs`
//! (a real kernel source using `T::shared_alloc`/`shared_store_index`/
//! `shared_barrier`/`shared_trans`/`shared_load_index`) through the real
//! MIR-codegen dispatch path (`ops/terminator.rs` + `ops/triton/tensor.rs`),
//! not a hand-built melior module.
//!
//! `test_indexed_shared_memory.rs` (a sibling test) already proved the
//! `tt.shared_*` marker ops + `tritongpu-lower-indexed-shared-memory` pass
//! work when hand-built directly at the MLIR level. This test instead proves
//! the front-end half: that a real Rust kernel calling the `Triton` trait's
//! `shared_*` methods gets routed, through the normal `rustc_driver` MIR
//! pipeline and the dispatch table, into handlers that emit those same
//! marker ops — closing the loop from source to PTX.

#![feature(rustc_private)]

use std::env;
use std::path::{Path, PathBuf};

use rustc_driver::{Callbacks, run_compiler};
use rustc_interface::interface;
use tracing::{debug, info};

struct MlirBackendCallbacks;

impl Callbacks for MlirBackendCallbacks {
    fn config(&mut self, config: &mut interface::Config) {
        config.make_codegen_backend = Some(Box::new(|_sess: &rustc_session::Session| {
            rustc_codegen_llvm::mlir::MlirCodegenBackend::new()
        }));
    }
}

#[derive(Debug, Clone, Default)]
pub struct LlvmCompiler {}

impl LlvmCompiler {
    pub fn new() -> Self {
        Self {}
    }

    pub fn compile(
        &self,
        filename: &Path,
        target: &str,
        output_name: &str,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let working_dir = PathBuf::from("/tmp");

        let mut callbacks = MlirBackendCallbacks;
        let exe_name = "/home/arshadm/.cargo/bin/rustc".to_string();
        let output_path = working_dir.join(format!("kernel-{output_name}.asm"));
        let output = format!("-o{}", output_path.display());
        let build_type = "-Copt-level=3".to_string();
        let panic_abort = "-Cpanic=abort".to_string();
        let target = format!("--target={}", target);
        let crate_type = "--crate-type=lib".to_string();
        let overflow_checks = "-C".to_string();
        let overflow_checks_off = "overflow-checks=off".to_string();
        let frontend = "--frontend=triton".to_string();

        info!("Working directory: {}", working_dir.display());
        info!("Target: {}", target);
        info!("Output: {}", output);
        debug!(
            "Rustc command: {} {} {} {} {}",
            exe_name,
            filename.display(),
            output,
            target,
            crate_type
        );

        unsafe {
            env::set_var("CFG_VERSION", "tg-1.90.0");
        }

        let args = vec![
            exe_name,
            filename.display().to_string(),
            build_type,
            panic_abort,
            output,
            target,
            crate_type,
            overflow_checks,
            overflow_checks_off,
            frontend,
        ];

        run_compiler(&args, &mut callbacks);

        Ok(output_path)
    }
}

#[cfg(test)]
mod tests {
    use tracing_subscriber::{EnvFilter, fmt};

    use super::*;

    #[test]
    fn test_indexed_shared_memory_kernel() -> Result<(), Box<dyn std::error::Error>> {
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let kernel =
            env::current_dir().unwrap().join("tests/data/triton_indexed_shared_memory.rs");
        let target = "nvptx64-nvidia-cuda";

        println!("Compiling indexed shared-memory kernel with target: {}", kernel.display());
        let output_path = compiler.compile(&kernel, target, "test_indexed_shared_memory")?;

        let ptx = std::fs::read_to_string(&output_path)
            .map_err(|e| format!("reading compiled PTX at {:?}: {e}", output_path))?;
        assert!(!ptx.trim().is_empty(), "expected non-empty PTX output at {:?}", output_path);
        assert!(
            ptx.contains(".shared"),
            "expected a shared-memory buffer in the PTX, got:\n{ptx}"
        );
        println!("PTX output ({} bytes) at {:?}", ptx.len(), output_path);

        Ok(())
    }
}
