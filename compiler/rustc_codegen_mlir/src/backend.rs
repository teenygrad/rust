use std::any::Any;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use rustc_ast::expand::allocator::AllocatorKind;
use rustc_codegen_ssa::back::lto::{SerializedModule, ThinModule};
use rustc_codegen_ssa::back::write::{
    CodegenContext, FatLtoInput, ModuleConfig, TargetMachineFactoryConfig, TargetMachineFactoryFn,
};
use rustc_codegen_ssa::traits::*;
use rustc_codegen_ssa::{CodegenResults, CompiledModule, ModuleCodegen, TargetConfig};
use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::DiagCtxtHandle;
use rustc_metadata::EncodedMetadata;
use rustc_middle::dep_graph;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::ty::TyCtxt;
use rustc_middle::util::Providers;
use rustc_session::Session;
use rustc_session::config::{OptLevel, OutputFilenames, PrintKind, PrintRequest};
use rustc_span::Symbol;

use crate::context::CodegenCx;
use crate::module::ModuleMlir;

#[derive(Clone)]
pub struct MlirCodegenBackend(());

impl ExtraBackendMethods for MlirCodegenBackend {
    fn codegen_allocator<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        module_name: &str,
        _kind: AllocatorKind,
        _alloc_error_handler_kind: AllocatorKind,
    ) -> ModuleMlir {
        // TODO: Implement allocator codegen
        ModuleMlir::new(tcx, module_name)
    }

    fn compile_codegen_unit(
        &self,
        tcx: TyCtxt<'_>,
        cgu_name: Symbol,
    ) -> (ModuleCodegen<ModuleMlir>, u64) {
        let start_time = Instant::now();

        let dep_node = tcx.codegen_unit(cgu_name).codegen_dep_node(tcx);
        let (module, _) = tcx.dep_graph.with_task(
            dep_node,
            tcx,
            cgu_name,
            module_codegen,
            Some(dep_graph::hash_result),
        );
        let time_to_codegen = start_time.elapsed();

        // We assume that the cost to run LLVM on a CGU is proportional to
        // the time we needed for codegenning it.
        let cost = time_to_codegen.as_nanos() as u64;

        fn module_codegen(tcx: TyCtxt<'_>, cgu_name: Symbol) -> ModuleCodegen<ModuleMlir> {
            let cgu = tcx.codegen_unit(cgu_name);
            let _prof_timer =
                tcx.prof.generic_activity_with_arg_recorder("codegen_module", |recorder| {
                    recorder.record_arg(cgu_name.to_string());
                    recorder.record_arg(cgu.size_estimate().to_string());
                });
            // Instantiate monomorphizations without filling out definitions yet...
            let mut llvm_module = ModuleMlir::new(tcx, cgu_name.as_str());
            {
                let mut cx = CodegenCx::new(tcx, cgu, &llvm_module);
                let mono_items = cx.codegen_unit.items_in_deterministic_order(cx.tcx);
                for &(mono_item, data) in &mono_items {
                    mono_item.predefine::<Builder<'_, '_, '_>>(
                        &mut cx,
                        cgu_name.as_str(),
                        data.linkage,
                        data.visibility,
                    );
                }

                // ... and now that we have everything pre-defined, fill out those definitions.
                for &(mono_item, item_data) in &mono_items {
                    mono_item.define::<Builder<'_, '_, '_>>(&mut cx, cgu_name.as_str(), item_data);
                }

                // If this codegen unit contains the main function, also create the
                // wrapper here
                if let Some(entry) =
                    maybe_create_entry_wrapper::<Builder<'_, '_, '_>>(&cx, cx.codegen_unit)
                {
                    let attrs = attributes::sanitize_attrs(&cx, SanitizerSet::empty());
                    attributes::apply_to_llfn(entry, llvm::AttributePlace::Function, &attrs);
                }

                // Finalize code coverage by injecting the coverage map. Note, the coverage map will
                // also be added to the `llvm.compiler.used` variable, created next.
                if cx.sess().instrument_coverage() {
                    cx.coverageinfo_finalize();
                }

                // Create the llvm.used and llvm.compiler.used variables.
                if !cx.used_statics.is_empty() {
                    cx.create_used_variable_impl(c"llvm.used", &cx.used_statics);
                }
                if !cx.compiler_used_statics.is_empty() {
                    cx.create_used_variable_impl(c"llvm.compiler.used", &cx.compiler_used_statics);
                }

                // Run replace-all-uses-with for statics that need it. This must
                // happen after the llvm.used variables are created.
                for &(old_g, new_g) in cx.statics_to_rauw().borrow().iter() {
                    unsafe {
                        llvm::LLVMReplaceAllUsesWith(old_g, new_g);
                        llvm::LLVMDeleteGlobal(old_g);
                    }
                }

                // Finalize debuginfo
                if cx.sess().opts.debuginfo != DebugInfo::None {
                    cx.debuginfo_finalize();
                }
            }

            ModuleCodegen::new_regular(cgu_name.to_string(), llvm_module)
        }

        (module, cost)
    }

    fn target_machine_factory(
        &self,
        _sess: &Session,
        _optlvl: OptLevel,
        _target_features: &[String],
    ) -> TargetMachineFactoryFn<Self> {
        // TODO: Implement target machine factory
        Arc::new(|_config: TargetMachineFactoryConfig| Ok(()))
    }

    fn spawn_named_thread<F, T>(
        _time_trace: bool,
        name: String,
        f: F,
    ) -> std::io::Result<std::thread::JoinHandle<T>>
    where
        F: FnOnce() -> T,
        F: Send + 'static,
        T: Send + 'static,
    {
        std::thread::Builder::new().name(name).spawn(f)
    }
}

impl WriteBackendMethods for MlirCodegenBackend {
    type Module = ModuleMlir;
    type ModuleBuffer = ModuleBuffer;
    type TargetMachine = ();
    type TargetMachineError = String;
    type ThinData = ThinData;
    type ThinBuffer = ThinBuffer;

    fn print_pass_timings(&self) {
        // TODO: Implement pass timings
    }

    fn print_statistics(&self) {
        // TODO: Implement statistics
    }

    #[allow(unreachable_code)]
    fn run_and_optimize_fat_lto(
        _cgcx: &CodegenContext<Self>,
        _exported_symbols_for_lto: &[String],
        _each_linked_rlib_for_lto: &[PathBuf],
        mut modules: Vec<FatLtoInput<Self>>,
    ) -> ModuleCodegen<Self::Module> {
        // TODO: Implement fat LTO
        // For now, just return the first module
        if let Some(first) = modules.pop() {
            match first {
                FatLtoInput::InMemory(module) => module,
                FatLtoInput::Serialized { .. } => {
                    panic!("Serialized modules not yet supported in fat LTO")
                }
            }
        } else {
            panic!("No modules provided for fat LTO")
        }
    }

    fn run_thin_lto(
        _cgcx: &CodegenContext<Self>,
        _exported_symbols_for_lto: &[String],
        _each_linked_rlib_for_lto: &[PathBuf],
        _modules: Vec<(String, Self::ThinBuffer)>,
        cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>,
    ) -> (Vec<ThinModule<Self>>, Vec<WorkProduct>) {
        // TODO: Implement thin LTO
        // For now, return empty vectors
        (Vec::new(), cached_modules.into_iter().map(|(_, wp)| wp).collect())
    }

    fn optimize(
        _cgcx: &CodegenContext<Self>,
        _dcx: DiagCtxtHandle<'_>,
        _module: &mut ModuleCodegen<Self::Module>,
        _config: &ModuleConfig,
    ) {
        // TODO: Implement optimization
    }

    fn optimize_thin(
        _cgcx: &CodegenContext<Self>,
        _thin: ThinModule<Self>,
    ) -> ModuleCodegen<Self::Module> {
        // TODO: Implement thin optimization
        // For now, create a dummy module
        panic!("Thin LTO optimization not yet implemented")
    }

    #[allow(unreachable_code)]
    fn codegen(
        cgcx: &CodegenContext<Self>,
        module: ModuleCodegen<Self::Module>,
        _config: &ModuleConfig,
    ) -> CompiledModule {
        let frontend = cgcx.opts.frontend.expect("frontend not set");
        todo!("codegen - mlir - {:?}", frontend);

        // TODO: Implement actual codegen
        CompiledModule {
            name: module.name,
            kind: module.kind,
            object: None,
            dwarf_object: None,
            bytecode: None,
            assembly: None,
            llvm_ir: None,
            links_from_incr_cache: Vec::new(),
        }
    }

    fn prepare_thin(module: ModuleCodegen<Self::Module>) -> (String, Self::ThinBuffer) {
        // TODO: Implement thin preparation
        (module.name, ThinBuffer::new())
    }

    fn serialize_module(module: ModuleCodegen<Self::Module>) -> (String, Self::ModuleBuffer) {
        // TODO: Implement module serialization
        (module.name, ModuleBuffer::new())
    }
}

impl MlirCodegenBackend {
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> Box<dyn CodegenBackend> {
        Box::new(MlirCodegenBackend(()))
    }
}

impl CodegenBackend for MlirCodegenBackend {
    fn locale_resource(&self) -> &'static str {
        ""
    }

    fn init(&self, _sess: &Session) {
        // TODO: Initialize MLIR backend
    }

    fn provide(&self, providers: &mut Providers) {
        providers.global_backend_features = |_tcx, ()| vec![];
    }

    fn print(&self, req: &PrintRequest, out: &mut String, _sess: &Session) {
        use std::fmt::Write;
        match req.kind {
            PrintKind::TargetFeatures => {
                writeln!(out, "Available target features:").unwrap();
                writeln!(out, "    (none)").unwrap();
                writeln!(out).unwrap();
            }
            _ => {
                // Default: do nothing
            }
        }
    }

    fn print_passes(&self) {
        // TODO: Implement pass printing
    }

    fn print_version(&self) {
        println!("MLIR codegen backend (basic implementation)");
    }

    fn target_config(&self, _sess: &Session) -> TargetConfig {
        TargetConfig {
            target_features: vec![],
            unstable_target_features: vec![],
            has_reliable_f16: false,
            has_reliable_f16_math: false,
            has_reliable_f128: false,
            has_reliable_f128_math: false,
        }
    }

    fn codegen_crate<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Box<dyn Any> {
        Box::new(rustc_codegen_ssa::base::codegen_crate(
            MlirCodegenBackend(()),
            tcx,
            "generic".to_string(),
        ))
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        _outputs: &OutputFilenames,
    ) -> (CodegenResults, FxIndexMap<WorkProductId, WorkProduct>) {
        let (codegen_results, work_products) = ongoing_codegen
            .downcast::<rustc_codegen_ssa::back::write::OngoingCodegen<MlirCodegenBackend>>()
            .expect("Expected MlirCodegenBackend's OngoingCodegen, found Box<Any>")
            .join(sess);

        (codegen_results, work_products)
    }

    fn link(
        &self,
        sess: &Session,
        codegen_results: CodegenResults,
        metadata: EncodedMetadata,
        outputs: &OutputFilenames,
    ) {
        use rustc_codegen_ssa::back::archive::ArArchiveBuilderBuilder;
        use rustc_codegen_ssa::back::link::link_binary;

        // Run the linker on any artifacts that resulted from codegen.
        link_binary(sess, &ArArchiveBuilderBuilder, codegen_results, metadata, outputs);
    }
}

// Placeholder types for ModuleBuffer and ThinBuffer

pub struct ModuleBuffer {
    data: Vec<u8>,
}

impl ModuleBuffer {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
}

impl rustc_codegen_ssa::traits::ModuleBufferMethods for ModuleBuffer {
    fn data(&self) -> &[u8] {
        &self.data
    }
}

pub struct ThinBuffer {
    data: Vec<u8>,
}

impl ThinBuffer {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
}

impl rustc_codegen_ssa::traits::ThinBufferMethods for ThinBuffer {
    fn data(&self) -> &[u8] {
        &self.data
    }
}

pub struct ThinData {
    // TODO: Add actual thin data fields
}

unsafe impl Send for ThinData {}
unsafe impl Sync for ThinData {}

// Export the backend entry point
#[unsafe(no_mangle)]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    MlirCodegenBackend::new()
}
