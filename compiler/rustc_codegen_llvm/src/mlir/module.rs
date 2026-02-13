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

use std::ffi::CStr;

use rustc_codegen_ssa::back::write::CodegenContext;
use rustc_errors::DiagCtxtHandle;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::Frontend;

use crate::mlir::backend::MlirCodegenBackend;
use crate::mlir::ffi::{
    MLIRContext, MLIRRustContextCreate, MLIRRustInitTriton, MLIRRustModuleBuilderCreate,
    MLIRRustModuleCreate, MLIRRustResult, ModuleOp,
};

/// Represents an MLIR module during codegen
pub struct MlirModule {
    pub name: String,
    pub(crate) mlcx: &'static mut MLIRContext,
    pub(crate) llmod_raw: *const ModuleOp,
}

unsafe impl Send for MlirModule {}
unsafe impl Sync for MlirModule {}

impl Drop for MlirModule {
    fn drop(&mut self) {
        todo!("Implement MlirModule drop");
    }
}

impl MlirModule {
    pub fn new(_tcx: TyCtxt<'_>, mod_name: &str) -> Self {
        let mlir_context = unsafe { MLIRRustContextCreate() };
        let builder = unsafe { MLIRRustModuleBuilderCreate(mlir_context) };
        let module = unsafe { MLIRRustModuleCreate(builder) };

        Self { name: mod_name.to_string(), mlcx: mlir_context, llmod_raw: module }
    }

    pub fn parse(
        _cgcx: &CodegenContext<MlirCodegenBackend>,
        name: &CStr,
        _buffer: &[u8],
        _dcx: DiagCtxtHandle<'_>,
    ) -> Self {
        let mlir_context = unsafe { MLIRRustContextCreate() };
        let builder = unsafe { MLIRRustModuleBuilderCreate(mlir_context) };
        let module = unsafe { MLIRRustModuleCreate(builder) };

        Self { name: name.to_string_lossy().to_string(), mlcx: mlir_context, llmod_raw: module }
    }

    pub fn set_llmod(&mut self, llmod: *const ModuleOp) {
        self.llmod_raw = llmod;
    }

    pub fn llmod(&self) -> &ModuleOp {
        unsafe { &*self.llmod_raw }
    }
}
