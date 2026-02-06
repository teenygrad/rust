// Copyright (C) 2026 Teenygrad. All rights reserved.

//! Bindings to the MLIR C API and our own `extern "C"` wrapper functions
//! around MLIR functionality (`MLIRRust*`).

#![allow(non_camel_case_types)]

// Opaque pointer types
unsafe extern "C" {
    pub(crate) type MLIRContext;
    pub(crate) type OpBuilder;
    pub(crate) type ModuleOp;
}

#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
#[allow(dead_code)] // Variants constructed by C++.
pub(crate) enum MLIRRustResult {
    Success,
    Failure,
}

#[link(name = "llvm-wrapper", kind = "static")]
unsafe extern "C" {
    pub(crate) fn MLIRRustContextCreate() -> &'static mut MLIRContext;

    pub(crate) fn MLIRRustInitTriton(context: &MLIRContext) -> MLIRRustResult;

    pub(crate) fn MLIRRustModuleBuilderCreate(context: &MLIRContext) -> &'static mut OpBuilder;

    pub(crate) fn MLIRRustModuleCreate(builder: &OpBuilder) -> &'static mut ModuleOp;

}
