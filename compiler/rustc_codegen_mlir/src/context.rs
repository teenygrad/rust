use core::borrow::Borrow;
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};

use rustc_abi::Size;
use rustc_middle::mir::mono::CodegenUnit;
use rustc_middle::ty::TyCtxt;

use crate::ModuleMlir;
use crate::ffi::{MLIRContext, ModuleOp, Type};

/// `TyCtxt` (and related cache datastructures) can't be move between threads.
/// However, there are various cx related functions which we want to be available to the builder and
/// other compiler pieces. Here we define a small subset which has enough information and can be
/// moved around more freely.
pub(crate) struct SCx<'ll> {
    pub llmod: &'ll ModuleOp,
    pub llcx: &'ll MLIRContext,
    pub isize_ty: &'ll Type,
}

impl<'ll> Borrow<SCx<'ll>> for FullCx<'ll, '_> {
    fn borrow(&self) -> &SCx<'ll> {
        &self.scx
    }
}

impl<'ll, 'tcx> Deref for FullCx<'ll, 'tcx> {
    type Target = SimpleCx<'ll>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.scx
    }
}

pub(crate) struct GenericCx<'ll, T: Borrow<SCx<'ll>>>(T, PhantomData<SCx<'ll>>);

impl<'ll, T: Borrow<SCx<'ll>>> Deref for GenericCx<'ll, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'ll, T: Borrow<SCx<'ll>>> DerefMut for GenericCx<'ll, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub(crate) type SimpleCx<'ll> = GenericCx<'ll, SCx<'ll>>;

/// There is one `CodegenCx` per codegen unit. Each one has its own LLVM
/// `llvm::Context` so that several codegen units may be processed in parallel.
/// All other LLVM data structures in the `CodegenCx` are tied to that `llvm::Context`.
pub(crate) type CodegenCx<'ll, 'tcx> = GenericCx<'ll, FullCx<'ll, 'tcx>>;

pub(crate) struct FullCx<'ll, 'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub scx: SimpleCx<'ll>,
    pub codegen_unit: &'tcx CodegenUnit<'tcx>,
}

impl<'ll, 'tcx> CodegenCx<'ll, 'tcx> {
    pub(crate) fn new(
        tcx: TyCtxt<'tcx>,
        codegen_unit: &'tcx CodegenUnit<'tcx>,
        llvm_module: &'ll ModuleMlir,
    ) -> Self {
        let (llcx, llmod) = (&*llvm_module.llcx, llvm_module.llmod());

        GenericCx(
            FullCx {
                tcx,
                scx: SimpleCx::new(llmod, llcx, tcx.data_layout.pointer_size()),
                codegen_unit,
            },
            PhantomData,
        )
    }
}

impl<'ll> SimpleCx<'ll> {
    pub(crate) fn new(llmod: &'ll ModuleOp, llcx: &'ll MLIRContext, pointer_size: Size) -> Self {
        let isize_ty = Type::ix_llcx(llcx, pointer_size.bits());
        Self(SCx { llmod, llcx, isize_ty }, PhantomData)
    }
}
