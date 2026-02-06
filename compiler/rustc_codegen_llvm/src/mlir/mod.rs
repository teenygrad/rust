// Copyright (C) 2026 Teenygrad. All rights reserved.

//! MLIR codegen backend implementation
//!
//! This module provides an alternative codegen backend using MLIR infrastructure.
//!
//! ## Target mechanism
//!
//! The MLIR backend is used with GPU and other non-CPU targets. Target selection is flexible:
//!
//! - **Builtin targets**: The `nvptx64-nvidia-cuda` target (and any other builtin that sets
//!   `default_codegen_backend: Some("mlir")` in `rustc_target::spec::targets`) uses the MLIR
//!   backend by default. Use `--target nvptx64-nvidia-cuda`; no need to pass `--codegen-backend=mlir`.
//!
//! - **Custom targets via JSON**: Define a target spec JSON file and set
//!   `"default-codegen-backend": "mlir"`. Then either:
//!   - Put `<triple>.json` in a directory listed in `RUST_TARGET_PATH`, or
//!   - Pass `--target /path/to/spec.json`.
//!     See `rustc_target::spec` for the full JSON schema.
//!
//! - **Adding new builtin targets**: Add a module under `rustc_target/src/spec/targets/` and
//!   register it in the `supported_targets!` macro in `rustc_target/src/spec/mod.rs`. Set
//!   `default_codegen_backend: Some("mlir".into())` in that target's `TargetOptions` to use
//!   the MLIR backend by default.
//!
//! ## Module Structure
//!
//! - `backend`: Main backend implementation (`MlirCodegenBackend`)
//! - `context`: Codegen context types for MLIR
//! - `ffi`: FFI bindings to MLIR/Triton C++ libraries
//! - `mir_visitor`: MIR traversal and logging utilities
//! - `module`: MLIR module representation
//! - `test_harness`: Test utilities for JIT and programmatic use

pub(crate) mod backend;
pub(crate) mod context;
pub(crate) mod ffi;
pub(crate) mod mir_visitor;
pub(crate) mod module;
pub mod test_harness;

pub use backend::MlirCodegenBackend;
pub use module::ModuleMlir;
