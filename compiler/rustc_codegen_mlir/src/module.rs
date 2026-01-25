use std::ffi::CStr;

use rustc_codegen_llvm::llvm::Context;
use rustc_codegen_ssa::back::write::CodegenContext;
use rustc_errors::DiagCtxtHandle;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::Frontend;

use crate::backend::MlirCodegenBackend;
use crate::mlir::ffi::{
    MLIRContext, MLIRRustContextCreate, MLIRRustInitTriton, MLIRRustModuleBuilderCreate,
    MLIRRustModuleCreate, MLIRRustResult, ModuleOp,
};

/// Represents an MLIR module during codegen
pub struct ModuleMlir {
    pub name: String,
    pub(crate) llcx: &'static mut Context,
    pub(crate) mlcx: &'static mut MLIRContext,
    pub(crate) llmod_raw: *const ModuleOp,
}

unsafe impl Send for ModuleMlir {}
unsafe impl Sync for ModuleMlir {}

impl ModuleMlir {
    pub fn new(tcx: TyCtxt<'_>, mod_name: &str) -> Self {
        let llvm_context = unsafe { MLIRRustContextCreate() };
        let mlir_context = unsafe { MLIRRustContextCreate() };
        let builder = unsafe { MLIRRustModuleBuilderCreate(llvm_context) };
        let module = unsafe { MLIRRustModuleCreate(builder) };
        let frontend = tcx.sess.opts.frontend.expect("frontend not set");

        let module = Self { name: mod_name.to_string(), llcx: llvm_context, llmod_raw: module };
        module.init_module(frontend);
        module
    }

    pub fn parse(
        cgcx: &CodegenContext<MlirCodegenBackend>,
        name: &CStr,
        _buffer: &[u8],
        _dcx: DiagCtxtHandle<'_>,
    ) -> Self {
        let context = unsafe { MLIRRustContextCreate() };
        let builder = unsafe { MLIRRustModuleBuilderCreate(context) };
        let module = unsafe { MLIRRustModuleCreate(builder) };
        let frontend = cgcx.opts.frontend.expect("frontend not set");

        let module =
            Self { name: name.to_string_lossy().to_string(), llcx: context, llmod_raw: module };
        module.init_module(frontend);
        module
    }

    pub fn set_llmod(&mut self, llmod: *const ModuleOp) {
        self.llmod_raw = llmod;
    }

    pub fn llmod(&self) -> &ModuleOp {
        unsafe { &*self.llmod_raw }
    }

    pub fn llcx(&self) -> &MLIRContext {
        self.llcx
    }

    fn init_module(&self, frontend: Frontend) {
        match frontend {
            Frontend::Triton => {
                let status = unsafe { MLIRRustInitTriton(self.llcx) };
                if status != MLIRRustResult::Success {
                    panic!("failed to initialize Triton");
                }
            }
        }
    }
}
