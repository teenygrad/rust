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

use std::collections::HashMap;

use melior::ir::{Block, BlockLike, BlockRef, Location, Operation, RegionLike};
use melior::utility::register_all_llvm_translations;
use rustc_middle::mir::mono::MonoItem;
use rustc_middle::mir::{
    BasicBlock, BasicBlockData, CastKind, Local, NonDivergingIntrinsic, Operand, Place, Rvalue,
    Statement, StatementKind, Terminator,
};
use rustc_middle::ty::layout::MaybeResult;
use rustc_middle::ty::{Instance, Ty, TyCtxt, TypingEnv};
use rustc_mlir::load_all_dialects;
use rustc_mlir::triton::tt::FuncOperation;
use rustc_mlir::triton::{create_tt_func_with_divisibility, load_triton_dialect};

use crate::mlir::MlirModule;
use crate::mlir::codegen::Codegen;
use crate::mlir::codegen::types::TypeMapper;
use crate::mlir::errors::MlirError;

pub(crate) struct TritonCodegen<'a> {
    module: &'a MlirModule<'static>,
    type_mapper: TypeMapper<'a>,
}

impl<'a> TritonCodegen<'a> {
    pub(crate) fn new(module: &'a MlirModule<'static>) -> Self {
        let context = module.context();

        load_all_dialects(context);
        register_all_llvm_translations(context);
        load_triton_dialect(context);

        Self { module, type_mapper: TypeMapper::new(context) }
    }

    fn codegen_function<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        fn_ty: Ty<'tcx>,
        instance: &Instance<'tcx>,
    ) -> Result<(), MlirError> {
        let mut ssa_values: HashMap<rustc_middle::mir::Local, melior::ir::Value<'tcx, 'static>> =
            HashMap::new();

        // Downcast to a FnSig
        let fn_sig = fn_ty.fn_sig(tcx);
        let fn_sig = fn_sig.skip_binder(); // Remove late-bound lifetimes

        // Extract a friendly function name, preferring unmangled if possible
        let func_name = tcx.symbol_name(*instance).name;

        // Try to demangle using the Rust symbol demangling crate if available.
        // Since in rustc we don't always bring in the rustc-demangle crate, but
        // the symbol_name should generally be readable for non-generic items.
        // Otherwise, fallback to `def_path_str` (should give a crate-relative path).
        let friendly_name = if func_name.starts_with("_R") {
            // Looks like a Rust-mangled symbol. Try to show a better name.
            tcx.def_path_str(instance.def_id())
        } else {
            func_name.to_string()
        };

        eprintln!(
            "[DEBUG] TritonCodegen: function name: {} (raw symbol: {})",
            friendly_name, func_name
        );

        // Arguments
        let arg_types: Vec<_> =
            fn_sig.inputs().iter().map(|ty| self.type_mapper.map_type(&tcx, ty)).collect();

        // Result type
        let ret_types = self.type_mapper.map_type(&tcx, &fn_sig.output()).to_result().ok();
        let ret_types = ret_types.as_slice();

        // DEBUG output: print argument and result types
        eprintln!("[DEBUG] TritonCodegen: instance function signature (argument types):");
        for (i, arg_ty) in arg_types.iter().enumerate() {
            eprintln!("    arg[{}]: {}", i, arg_ty);
        }
        eprintln!(
            "[DEBUG] TritonCodegen: instance function signature (return type): {:?}",
            ret_types
        );

        // Iterate over MIR basic blocks and codegen each one
        let func_op = create_tt_func_with_divisibility(
            self.module.context(),
            Location::unknown(self.module.context()),
            func_name,
            &arg_types,
            ret_types,
            16,
        );

        let mir = tcx.instance_mir(instance.def);
        for (bb, bb_data) in mir.basic_blocks.iter_enumerated() {
            self.codegen_basic_block(tcx, bb, bb_data, &func_op, &mut ssa_values)?;
        }

        self.module.llmod().body().append_operation(func_op.into());

        Ok(())
    }

    fn codegen_basic_block<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        _bb: BasicBlock,
        bb_data: &BasicBlockData<'tcx>,
        func_op: &FuncOperation<'a>,
        ssa_values: &mut HashMap<rustc_middle::mir::Local, melior::ir::Value<'tcx, 'static>>,
    ) -> Result<(), MlirError> {
        // Create an empty MLIR block and append it to the function body region.
        // Block arguments will be added when argument-passing lowering is implemented.
        let mlir_block = Block::new(&[]);
        let mlir_block =
            func_op.body().expect("tt.func must have a body region").append_block(mlir_block);

        // Codegen each MIR statement in order.
        for stmt in &bb_data.statements {
            self.codegen_statement(tcx, stmt, &mlir_block, ssa_values)?;
        }

        // Codegen the block terminator.
        self.codegen_terminator(tcx, bb_data.terminator(), &mlir_block)?;

        Ok(())
    }

    fn codegen_statement<'tcx, 'blk>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        stmt: &Statement<'tcx>,
        mlir_block: &BlockRef<'a, 'blk>,
        ssa_values: &mut HashMap<rustc_middle::mir::Local, melior::ir::Value<'tcx, 'static>>,
    ) -> Result<(), MlirError> {
        match &stmt.kind {
            StatementKind::Assign(assign) => {
                let (place, rvalue) = assign.as_ref();
                println!(
                    "[DEBUG] TritonCodegen::codegen_statement: Assign: {:?}, {:?} {:?}",
                    stmt, place, rvalue
                );
                self.codegen_assign(tcx, place, rvalue, mlir_block, ssa_values)
            }
            StatementKind::SetDiscriminant { place, variant_index } => {
                self.codegen_set_discriminant(tcx, place, *variant_index, mlir_block)
            }
            StatementKind::StorageLive(local) => self.codegen_storage_live(tcx, *local, mlir_block),
            StatementKind::StorageDead(local) => self.codegen_storage_dead(tcx, *local, mlir_block),
            StatementKind::Intrinsic(intrinsic) => {
                self.codegen_intrinsic(tcx, intrinsic, mlir_block)
            }
            // Runtime no-ops or analysis-only statements that require no codegen.
            StatementKind::Nop
            | StatementKind::ConstEvalCounter
            | StatementKind::FakeRead(_)
            | StatementKind::PlaceMention(_)
            | StatementKind::AscribeUserType(..)
            | StatementKind::Coverage(_)
            | StatementKind::BackwardIncompatibleDropHint { .. }
            | StatementKind::Retag(..) => Ok(()),
        }
    }

    fn codegen_assign<'tcx, 'blk>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        place: &Place<'tcx>,
        rvalue: &Rvalue<'tcx>,
        mlir_block: &BlockRef<'a, 'blk>,
        ssa_values: &mut HashMap<rustc_middle::mir::Local, melior::ir::Value<'tcx, 'static>>,
    ) -> Result<(), MlirError> {
        let _rvalue_op: Operation<'tcx> = match rvalue {
            Rvalue::Use(operand) => todo!("Use operand: {:?}", operand),
            Rvalue::Repeat(operand, _) => todo!("Repeat operand: {:?}", operand),
            Rvalue::Ref(region, borrow_kind, place) => todo!(
                "Ref region: {:?}, borrow_kind: {:?}, place: {:?}",
                region,
                borrow_kind,
                place
            ),
            Rvalue::ThreadLocalRef(def_id) => todo!("ThreadLocalRef def_id: {:?}", def_id),
            Rvalue::RawPtr(raw_ptr_kind, place) => {
                todo!("RawPtr raw_ptr_kind: {:?}, place: {:?}", raw_ptr_kind, place)
            }
            Rvalue::Cast(cast_kind, operand, ty) => {
                println!("Cast cast_kind: {:?}, operand: {:?}, ty: {:?}", cast_kind, operand, ty);
                self.codegen_cast(tcx, cast_kind, operand, ty, mlir_block, ssa_values)
            }
            Rvalue::BinaryOp(bin_op, _) => todo!("BinaryOp bin_op: {:?}", bin_op),
            Rvalue::NullaryOp(null_op) => todo!("NullaryOp null_op: {:?}", null_op),
            Rvalue::UnaryOp(un_op, operand) => {
                todo!("UnaryOp un_op: {:?}, operand: {:?}", un_op, operand)
            }
            Rvalue::Discriminant(place) => todo!("Discriminant place: {:?}", place),
            Rvalue::Aggregate(aggregate_kind, index_vec) => {
                todo!("Aggregate aggregate_kind: {:?}, index_vec: {:?}", aggregate_kind, index_vec)
            }
            Rvalue::ShallowInitBox(operand, ty) => {
                todo!("ShallowInitBox operand: {:?}, ty: {:?}", operand, ty)
            }
            Rvalue::CopyForDeref(place) => todo!("CopyForDeref place: {:?}", place),
            Rvalue::WrapUnsafeBinder(operand, ty) => {
                todo!("WrapUnsafeBinder operand: {:?}, ty: {:?}", operand, ty)
            }
        };

        // todo!("[TODO] TritonCodegen::codegen_assign: {:?} {:?}", place, rvalue)
        Ok(())
    }

    fn codegen_set_discriminant<'tcx, 'blk>(
        &mut self,
        _tcx: TyCtxt<'tcx>,
        _place: &Place<'tcx>,
        _variant_index: rustc_abi::VariantIdx,
        _mlir_block: &BlockRef<'a, 'blk>,
    ) -> Result<(), MlirError> {
        todo!("[TODO] TritonCodegen::codegen_set_discriminant")
    }

    fn codegen_storage_live<'tcx, 'blk>(
        &mut self,
        _tcx: TyCtxt<'tcx>,
        _local: Local,
        _mlir_block: &BlockRef<'a, 'blk>,
    ) -> Result<(), MlirError> {
        // NO-OP: In the context of Triton and MLIR, storage live is a no-op.
        Ok(())
    }

    fn codegen_storage_dead<'tcx, 'blk>(
        &mut self,
        _tcx: TyCtxt<'tcx>,
        _local: Local,
        _mlir_block: &BlockRef<'a, 'blk>,
    ) -> Result<(), MlirError> {
        // NO-OP: In the context of Triton and MLIR, storage dead is a no-op.
        Ok(())
    }

    fn codegen_intrinsic<'tcx, 'blk>(
        &mut self,
        _tcx: TyCtxt<'tcx>,
        _intrinsic: &NonDivergingIntrinsic<'tcx>,
        _mlir_block: &BlockRef<'a, 'blk>,
    ) -> Result<(), MlirError> {
        todo!("[TODO] TritonCodegen::codegen_intrinsic")
    }

    fn codegen_terminator<'tcx, 'blk>(
        &mut self,
        _tcx: TyCtxt<'tcx>,
        _terminator: &Terminator<'tcx>,
        _mlir_block: &BlockRef<'a, 'blk>,
    ) -> Result<(), MlirError> {
        todo!("[TODO] TritonCodegen::codegen_terminator")
    }

    fn codegen_cast<'tcx, 'blk>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        cast_kind: &CastKind,
        operand: &Operand<'tcx>,
        ty: &Ty<'tcx>,
        mlir_block: &BlockRef<'a, 'blk>,
        ssa_values: &mut HashMap<rustc_middle::mir::Local, melior::ir::Value<'tcx, 'static>>,
    ) -> Operation<'tcx> {
        match cast_kind {
            CastKind::PointerExposeProvenance => todo!("PointerExposeProvenance"),
            CastKind::PointerWithExposedProvenance => self
                .codegen_pointer_with_exposed_provenance(tcx, operand, ty, mlir_block, ssa_values),
            CastKind::PointerCoercion(pointer_coercion, coercion_source) => todo!(
                "PointerCoercion pointer_coercion: {:?}, coercion_source: {:?}",
                pointer_coercion,
                coercion_source
            ),
            CastKind::IntToInt => todo!("IntToInt"),
            CastKind::FloatToInt => todo!("FloatToInt"),
            CastKind::FloatToFloat => todo!("FloatToFloat"),
            CastKind::IntToFloat => todo!("IntToFloat"),
            CastKind::PtrToPtr => todo!("PtrToPtr"),
            CastKind::FnPtrToPtr => todo!("FnPtrToPtr"),
            CastKind::Transmute => todo!("Transmute"),
            CastKind::Subtype => todo!("Subtype"),
        }
    }

    fn codegen_pointer_with_exposed_provenance<'tcx, 'blk>(
        &mut self,
        _tcx: TyCtxt<'tcx>,
        _operand: &Operand<'tcx>,
        _ty: &Ty<'tcx>,
        _mlir_block: &BlockRef<'a, 'blk>,
        _ssa_values: &mut HashMap<rustc_middle::mir::Local, melior::ir::Value<'tcx, 'static>>,
    ) -> Operation<'tcx> {
        todo!("[TODO] TritonCodegen::codegen_pointer_with_exposed_provenance")
    }
}

impl<'a> Codegen for TritonCodegen<'a> {
    fn codegen<'tcx>(&mut self, tcx: TyCtxt<'tcx>, item: &MonoItem<'tcx>) -> Result<(), MlirError> {
        match item {
            MonoItem::Fn(instance) => {
                let fn_ty = instance.ty(tcx, TypingEnv::fully_monomorphized());
                let is_fn_ty = matches!(
                    fn_ty.kind(),
                    rustc_middle::ty::TyKind::FnDef(..) | rustc_middle::ty::TyKind::FnPtr(_, _)
                );

                if !is_fn_ty {
                    todo!(
                        "[DEBUG] TritonCodegen: instance.ty(tcx) is not a function type: {:?}",
                        fn_ty
                    );
                }

                self.codegen_function(tcx, fn_ty, instance)
            }
            MonoItem::Static(_def_id) => {
                // TODO: Implement Triton codegen for statics
                todo!()
            }
            MonoItem::GlobalAsm(_item_id) => {
                // TODO: Implement Triton codegen for global asm
                todo!()
            }
        }
    }
}
