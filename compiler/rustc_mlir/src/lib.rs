//! Rust bindings for MLIR and Triton dialects.
//!
//! This crate provides:
//! - FFI bindings to the mlir-wrapper C++ library for Triton-specific types
//! - Re-exports of melior for general MLIR construction
//! - Helper types for building Triton IR from Rust
//!
//! # Architecture
//!
//! The crate is structured in layers:
//! 1. `ffi` - Raw C FFI bindings to mlir-wrapper
//! 2. `triton` - Safe Rust wrappers around Triton types
//! 3. `builder` - High-level builder API using melior's OperationBuilder
//!
//! # Example
//!
//! ```ignore
//! use rustc_mlir::{context::Context, triton};
//!
//! let context = Context::new();
//! triton::register_dialects(&context);
//!
//! // Use melior's OperationBuilder for construction
//! let module = context.create_module("my_kernel");
//! ```

#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]

pub mod ffi;
pub mod triton;
pub mod context;
pub mod builder;

// Re-export melior for convenience
pub use melior;

/// Error type for MLIR operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    #[error("MLIR operation failed")]
    OperationFailed,

    #[error("Triton dialects not available (compiled without TRITON_ENABLED)")]
    TritonNotAvailable,

    #[error("Invalid type: {0}")]
    InvalidType(String),

    #[error("Invalid attribute: {0}")]
    InvalidAttribute(String),

    #[error("Module verification failed")]
    VerificationFailed,

    #[error("Parse error: {0}")]
    ParseError(String),
}

/// Result type for MLIR operations
pub type Result<T> = std::result::Result<T, Error>;
