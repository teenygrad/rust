/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

//! MLIR codegen backend implementation
//!
//! This module provides an alternative codegen backend using MLIR infrastructure.
//!
//! ## Target mechanism
//!
//! The MLIR backend is used with GPU and other non-CPU targets. Target selection follows the
//! same split rustc uses everywhere: the `--target` triple's *architecture* picks which
//! hardware backend (Cuda/Riscv, see `rustc_mlir::ffi::TargetBackend`) a compile uses, and
//! `-C target-cpu` picks the specific model/capability within it (see [`target::resolve`]).
//!
//! - **Builtin targets**: `nvptx64-nvidia-cuda` (arch `Nvptx64` -> Cuda backend, `-C
//!   target-cpu=sm_90` etc.) and `riscv64-generic` (arch `RiscV64`/`RiscV32` -> Riscv backend,
//!   `-C target-cpu=spacemit-k3` etc., currently a stub -- see `RiscvBackend.h`) both set
//!   `default_codegen_backend: Some("mlir")` in `rustc_target::spec::targets`, so `--target
//!   nvptx64-nvidia-cuda` / `--target riscv64-generic` alone select this backend; no need to
//!   also pass `--codegen-backend=mlir`.
//!
//! - **Custom targets via JSON**: Define a target spec JSON file with an `arch` this backend
//!   recognizes (see [`target::resolve`]) and set `"default-codegen-backend": "mlir"`. Then
//!   either put `<triple>.json` in a directory listed in `RUST_TARGET_PATH`, or pass `--target
//!   /path/to/spec.json`. See `rustc_target::spec` for the full JSON schema.
//!
//! - **Adding a new arch**: extend the match in [`target::resolve`] (and add a corresponding
//!   `rustc_mlir::ffi::TargetBackend` variant / C++ backend if it's a genuinely new hardware
//!   backend, not just a new triple for an existing one).
//!
//! ## Module Structure
//!
//! - `backend`: Main backend implementation (`MlirCodegenBackend`)
//! - `codegen`: Codegen trait and implementation
//! - `context`: Codegen context types for MLIR
//! - `error`: Error types for MLIR codegens
//! - `ffi`: FFI bindings to MLIR/Triton C++ libraries
//! - `mir_visitor`: MIR traversal and logging utilities
//! - `module`: MLIR module representation
//! - `target`: `--target`/`-C target-cpu` -> `CompileOptions` resolution
//! - `test_harness`: Test utilities for JIT and programmatic use

pub(crate) mod backend;
pub(crate) mod codegen;
pub(crate) mod context;
pub(crate) mod errors;
pub(crate) mod ffi;
pub(crate) mod mir_visitor;
pub(crate) mod module;
pub(crate) mod target;

pub use backend::MlirCodegenBackend;
pub use module::MlirModule;

/// `tracing` target used for this backend's own diagnostic logging, scoped
/// separately from the rest of `rustc` so a caller can turn up verbosity for
/// just the MLIR pipeline (e.g. `RUSTC_LOG=rustc_codegen_llvm::mlir=debug`)
/// without drowning in unrelated compiler internals. At `debug`, each
/// pipeline stage's IR (ttir, ttgpuir, llir, llvmir, ptx/asm) is logged once
/// per stage; at `trace`, the Triton MLIR passes additionally print IR
/// before/after every individual pass (see [`module::MlirModule::new_with_capability`]
/// wiring this into `CompileOptions::debug`).
pub(crate) const LOG_TARGET: &str = "rustc_codegen_llvm::mlir";
