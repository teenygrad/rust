//! Bindings to the LLVM-C API (`LLVM*`), and to our own `extern "C"` wrapper
//! functions around the unstable LLVM C++ API (`LLVMRust*`).
//!
//! ## Passing pointer/length strings as `*const c_uchar` (PTR_LEN_STR)
//!
//! Normally it's a good idea for Rust-side bindings to match the corresponding
//! C-side function declarations as closely as possible. But when passing `&str`
//! or `&[u8]` data as a pointer/length pair, it's more convenient to declare
//! the Rust-side pointer as `*const c_uchar` instead of `*const c_char`.
//! Both pointer types have the same ABI, and using `*const c_uchar` avoids
//! the need for an extra cast from `*const u8` on the Rust side.

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
