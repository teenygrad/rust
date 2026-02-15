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

// pub mod ffi;
// pub mod triton;
// pub mod context;
// pub mod builder;

// Re-export melior for convenience
pub use melior;

