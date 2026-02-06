// Copyright (C) 2026 Teenygrad. All rights reserved.

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
pub struct ModuleMlir {
    pub name: String,
    pub(crate) mlcx: &'static mut MLIRContext,
    pub(crate) llmod_raw: *const ModuleOp,
}

unsafe impl Send for ModuleMlir {}
unsafe impl Sync for ModuleMlir {}

impl ModuleMlir {
    pub fn new(tcx: TyCtxt<'_>, mod_name: &str) -> Self {
        let mlir_context = unsafe { MLIRRustContextCreate() };
        let builder = unsafe { MLIRRustModuleBuilderCreate(mlir_context) };
        let module = unsafe { MLIRRustModuleCreate(builder) };
        let frontend = tcx.sess.opts.frontend.expect("frontend not set");

        let module = Self { name: mod_name.to_string(), mlcx: mlir_context, llmod_raw: module };
        module.init_module(frontend);
        module
    }

    pub fn parse(
        cgcx: &CodegenContext<MlirCodegenBackend>,
        name: &CStr,
        _buffer: &[u8],
        _dcx: DiagCtxtHandle<'_>,
    ) -> Self {
        let mlir_context = unsafe { MLIRRustContextCreate() };
        let builder = unsafe { MLIRRustModuleBuilderCreate(mlir_context) };
        let module = unsafe { MLIRRustModuleCreate(builder) };
        let frontend = cgcx.opts.frontend.expect("frontend not set");

        let module = Self {
            name: name.to_string_lossy().to_string(),
            mlcx: mlir_context,
            llmod_raw: module,
        };
        module.init_module(frontend);
        module
    }

    pub fn set_llmod(&mut self, llmod: *const ModuleOp) {
        self.llmod_raw = llmod;
    }

    pub fn llmod(&self) -> &ModuleOp {
        unsafe { &*self.llmod_raw }
    }

    fn init_module(&self, frontend: Frontend) {
        match frontend {
            Frontend::Triton => {
                let status = unsafe { MLIRRustInitTriton(self.mlcx) };
                if status != MLIRRustResult::Success {
                    panic!("failed to initialize Triton");
                }
            }
        }
    }
}
