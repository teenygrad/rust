//! MLIR codegen backend implementation
//!
//! This module provides an alternative codegen backend using MLIR infrastructure.
//!
//! ## Module Structure
//!
//! - `backend`: Main backend implementation (`MlirCodegenBackend`)
//! - `context`: Codegen context types for MLIR
//! - `ffi`: FFI bindings to MLIR/Triton C++ libraries
//! - `mir_visitor`: MIR traversal and logging utilities
//! - `module`: MLIR module representation
//! - `test_harness`: Test utilities for JIT and programmatic use

pub(crate) mod ffi;
pub(crate) mod backend;
pub(crate) mod context;
pub(crate) mod mir_visitor;
pub(crate) mod module;
pub mod test_harness;

pub use backend::MlirCodegenBackend;
pub use module::ModuleMlir;
