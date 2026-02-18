/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
