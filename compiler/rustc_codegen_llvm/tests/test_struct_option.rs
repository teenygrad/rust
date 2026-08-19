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

//! teenyc-3af.1: compile `tests/data/triton_struct_option.rs` — a kernel that
//! constructs a local `Tile { tensor, mask: Option<BoolTensor> }`, field-
//! projects both `Some(mask)` and `None`, and copies the Option field to a
//! local before passing it into `T::load`/`T::store` — through the real
//! MIR-codegen path to PTX.
//!
//! Structs here are kernel-local only (intra-kernel, after inlining). They
//! are never kernel-entry ABI. Compiled at `-Copt-level=3` with MIR SROA
//! disabled so the `Tile` aggregate reaches codegen; without that, SROA
//! splits it into tensor/mask locals and the existing `option_table` path
//! already works. The gap is `tuple_fields: Vec<Value>` not composing with
//! `option_table` when the aggregate survives.

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
        // Keep opt-level=3 so Triton methods inline (same as production kernels).
        // Disable MIR SROA so `Tile { tensor, mask: Option<_> }` survives as an
        // aggregate into MLIR codegen — at opt-level=3 SROA splits it into
        // separate locals and the tuple_fields/option_table bug never fires.
        let build_type = "-Copt-level=3".to_string();
        let disable_sroa = "-Zmir-enable-passes=-ScalarReplacementOfAggregates".to_string();
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
            disable_sroa,
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
    fn test_struct_option_field_kernel() -> Result<(), Box<dyn std::error::Error>> {
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let kernel = env::current_dir().unwrap().join("tests/data/triton_struct_option.rs");
        let target = "nvptx64-nvidia-cuda";

        println!("Compiling struct+Option-field kernel with target: {}", kernel.display());
        let output_path = compiler.compile(&kernel, target, "test_struct_option")?;

        let ptx = std::fs::read_to_string(&output_path)
            .map_err(|e| format!("reading compiled PTX at {:?}: {e}", output_path))?;
        assert!(!ptx.trim().is_empty(), "expected non-empty PTX output at {:?}", output_path);
        // Masked load (Some) plus unmasked store (None) both reached codegen.
        assert!(
            ptx.contains("ld.global") || ptx.contains("ld.global.nc") || ptx.contains("ld.param"),
            "expected a global/param load in the PTX (masked tile.tensor load), got:\n{ptx}"
        );
        assert!(
            ptx.contains("st.global") || ptx.contains("st.param"),
            "expected a store in the PTX (tile.mask = None store), got:\n{ptx}"
        );
        println!("PTX output ({} bytes) at {:?}", ptx.len(), output_path);

        Ok(())
    }
}
