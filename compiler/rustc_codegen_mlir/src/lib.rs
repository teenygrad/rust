//! The MLIR codegen backend for Rust.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![allow(internal_features)]
#![feature(assert_matches)]
#![feature(impl_trait_in_assoc_type)]
#![feature(try_blocks)]
#![feature(rustdoc_internals)]
#![feature(extern_types)]

mod backend;
mod context;
mod mlir;
mod module;

pub use backend::MlirCodegenBackend;
pub use module::ModuleMlir;

// TODO: Add fluent messages when needed
// rustc_fluent_macro::fluent_messages! { "../messages.ftl" }
