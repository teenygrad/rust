use crate::spec::{
    Arch, Cc, CodeModel, LinkerFlavor, Lld, LlvmAbi, PanicStrategy, RelocModel, Target,
    TargetMetadata, TargetOptions,
};

/// Special-purpose target for compiling RISC-V vector kernels through the
/// `mlir` codegen backend's Triton/RISC-V path (see
/// `rustc_codegen_llvm::mlir`), analogous to how `nvptx64-nvidia-cuda`
/// exists purely to be compiled via that backend's CUDA path.
///
/// This is deliberately a separate triple from the stock `riscv64gc-*`
/// targets rather than a shared default: those targets compile ordinary
/// bare-metal/Linux RISC-V programs through the normal LLVM backend, and
/// must keep doing so unconditionally. Reusing them here (e.g. via
/// `default_codegen_backend`) would silently redirect every normal build for
/// that triple into this backend's (currently stub) RISC-V path instead.
pub(crate) fn target() -> Target {
    Target {
        data_layout: "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128".into(),
        metadata: TargetMetadata {
            description: Some(
                "RISC-V vector kernels via the `mlir` codegen backend's Triton/RISC-V path"
                    .into(),
            ),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        llvm_target: "riscv64".into(),
        pointer_width: 64,
        arch: Arch::RiscV64,

        options: TargetOptions {
            linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
            linker: Some("rust-lld".into()),
            llvm_abiname: LlvmAbi::Lp64d,
            // Not an LLVM `-mcpu` value: this backend's RISC-V path doesn't
            // go through LLVM's RISC-V codegen at all, so `-C target-cpu`
            // here names a Triton/RiscvBackend-side chip identifier instead
            // (e.g. `spacemit-k3`) -- see
            // `rustc_codegen_llvm::mlir::target::resolve_riscv`.
            cpu: "generic-rvv1.0".into(),
            max_atomic_width: Some(64),
            features: "+m,+a,+f,+d,+c,+v,+zicsr,+zifencei".into(),
            panic_strategy: PanicStrategy::Abort,
            relocation_model: RelocModel::Static,
            code_model: Some(CodeModel::Medium),
            emit_debug_gdb_scripts: false,
            eh_frame_header: false,
            // Route straight to the `mlir` codegen backend by default: unlike
            // the stock `riscv64gc-*` targets, this triple has no other
            // purpose.
            default_codegen_backend: Some("mlir".into()),
            ..Default::default()
        },
    }
}
