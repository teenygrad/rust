//! MLIR codegen backend implementation
//!
//! This module provides an alternative codegen backend using MLIR infrastructure.

pub(crate) mod ffi;
pub(crate) mod backend;
pub(crate) mod context;
pub(crate) mod module;

pub use backend::MlirCodegenBackend;
pub use module::ModuleMlir;
