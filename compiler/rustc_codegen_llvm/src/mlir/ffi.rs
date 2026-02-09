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
