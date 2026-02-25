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

use melior::dialect::ods::arith::ConstantOperation;
use melior::ir::operation::OperationLike;
use melior::ir::{
    Block, BlockLike, BlockRef, Location, Operation, RegionLike, TypeLike, ValueLike,
};
use melior::utility::register_all_llvm_translations;
use rustc_abi::{FieldIdx, VariantIdx};
use rustc_ast::UintTy;
use rustc_hir::def_id::DefId;
use rustc_index::IndexVec;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::mono::MonoItem;
use rustc_middle::mir::{
    AggregateKind, BasicBlock, BasicBlockData, CastKind, Const, ConstOperand, ConstValue, Local,
    NonDivergingIntrinsic, Operand, Place, Rvalue, Statement, StatementKind, Terminator,
};
use rustc_middle::ty::layout::MaybeResult;
use rustc_middle::ty::{
    EarlyBinder, GenericArg, Instance, Ty, TyCtxt, TyKind, TypingEnv, UserTypeAnnotationIndex,
};
use rustc_mlir::load_all_dialects;
use rustc_mlir::shared::arith::{Int, create_constant};
use rustc_mlir::shared::ub::create_ub_poison;
use rustc_mlir::triton::tt::FuncOperation;
use rustc_mlir::triton::{
    create_triton_ranked_tensor, create_tt_func_with_divisibility, create_tt_int_to_ptr_cast,
    create_tt_return, load_triton_dialect,
};

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
        let mut ssa_values: HashMap<rustc_middle::mir::Local, melior::ir::Value<'a, 'a>> =
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
            self.codegen_basic_block(tcx, instance, bb, bb_data, &func_op, &mut ssa_values)?;
        }

        self.module.llmod().body().append_operation(func_op.into());

        Ok(())
    }

    fn codegen_basic_block<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        _bb: BasicBlock,
        bb_data: &BasicBlockData<'tcx>,
        func_op: &FuncOperation<'a>,
        ssa_values: &mut HashMap<rustc_middle::mir::Local, melior::ir::Value<'a, 'a>>,
    ) -> Result<(), MlirError> {
        // Create an empty MLIR block and append it to the function body region.
        // Block arguments will be added when argument-passing lowering is implemented.
        let mlir_block = Block::new(&[]);
        let mlir_block =
            func_op.body().expect("tt.func must have a body region").append_block(mlir_block);

        // Codegen each MIR statement in order.
        for stmt in &bb_data.statements {
            self.codegen_statement(tcx, instance, stmt, &mlir_block, ssa_values)?;
        }

        // Codegen the block terminator.
        self.codegen_terminator(tcx, bb_data.terminator(), &mlir_block, ssa_values)?;

        Ok(())
    }

    fn codegen_statement<'tcx, 'blk>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        stmt: &Statement<'tcx>,
        mlir_block: &BlockRef<'a, 'blk>,
        ssa_values: &mut HashMap<rustc_middle::mir::Local, melior::ir::Value<'a, 'a>>,
    ) -> Result<(), MlirError> {
        match &stmt.kind {
            StatementKind::Assign(assign) => {
                let (place, rvalue) = assign.as_ref();
                println!(
                    "[DEBUG] TritonCodegen::codegen_statement: Assign: {:?}, {:?} {:?}",
                    stmt, place, rvalue
                );
                self.codegen_assign(tcx, instance, place, rvalue, mlir_block, ssa_values)
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
        instance: &Instance<'tcx>,
        place: &Place<'tcx>,
        rvalue: &Rvalue<'tcx>,
        mlir_block: &BlockRef<'a, 'blk>,
        ssa_values: &mut HashMap<rustc_middle::mir::Local, melior::ir::Value<'a, 'a>>,
    ) -> Result<(), MlirError> {
        match rvalue {
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
                let cast_op = self.codegen_cast(tcx, instance, cast_kind, operand, ty, mlir_block);
                let result = cast_op.result(0).unwrap();
                ssa_values.insert(place.local, result.into());
                mlir_block.append_operation(cast_op.into());
            }
            Rvalue::BinaryOp(bin_op, _) => todo!("BinaryOp bin_op: {:?}", bin_op),
            Rvalue::NullaryOp(null_op) => todo!("NullaryOp null_op: {:?}", null_op),
            Rvalue::UnaryOp(un_op, operand) => {
                todo!("UnaryOp un_op: {:?}, operand: {:?}", un_op, operand)
            }
            Rvalue::Discriminant(place) => todo!("Discriminant place: {:?}", place),
            Rvalue::Aggregate(aggregate_kind, index_vec) => {
                let aggregate_op =
                    self.codegen_aggregate(tcx, instance, aggregate_kind, index_vec, mlir_block);
                let result = aggregate_op.result(0).unwrap();
                ssa_values.insert(place.local, result.into());
                mlir_block.append_operation(aggregate_op);
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

    fn codegen_aggregate<'tcx, 'blk>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        aggregate_kind: &AggregateKind<'tcx>,
        _index_vec: &IndexVec<FieldIdx, Operand<'tcx>>,
        mlir_block: &BlockRef<'a, 'blk>,
    ) -> Operation<'a> {
        match aggregate_kind {
            AggregateKind::Array(ty) => todo!("Array ty: {:?}", ty),
            AggregateKind::Tuple => todo!("Tuple"),
            AggregateKind::Adt(
                def_id,
                variant_idx,
                raw_list,
                user_type_annotation_index,
                field_idx,
            ) => self.codegen_adt(
                tcx,
                instance,
                def_id,
                variant_idx,
                raw_list.as_slice(),
                user_type_annotation_index,
                field_idx,
                mlir_block,
            ),
            AggregateKind::Closure(def_id, raw_list) => {
                todo!("Closure def_id: {:?}, raw_list: {:?}", def_id, raw_list)
            }
            AggregateKind::Coroutine(def_id, raw_list) => {
                todo!("Coroutine def_id: {:?}, raw_list: {:?}", def_id, raw_list)
            }
            AggregateKind::CoroutineClosure(def_id, raw_list) => {
                todo!("CoroutineClosure def_id: {:?}, raw_list: {:?}", def_id, raw_list)
            }
            AggregateKind::RawPtr(ty, mutability) => {
                todo!("RawPtr ty: {:?}, mutability: {:?}", ty, mutability)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn codegen_adt<'tcx, 'blk>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        def_id: &DefId,
        variant_idx: &VariantIdx,
        raw_list: &[GenericArg<'tcx>],
        user_type_annotation_index: &Option<UserTypeAnnotationIndex>,
        field_idx: &Option<FieldIdx>,
        _mlir_block: &BlockRef<'a, 'blk>,
    ) -> Operation<'a> {
        let adt_def = tcx.adt_def(*def_id);
        let name = format!("{:?}", adt_def);

        // If the name of the ADT is tensor, then we create a poison operation.
        // This is because the tensor creation is part of the dsl dead code which
        // will be eliminated by the optimizer.
        if name == "triton::llvm::triton::tensor::Tensor" {
            let ty = instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                TypingEnv::fully_monomorphized(),
                EarlyBinder::bind(raw_list[0].expect_ty()),
            );
            let ty = self.type_mapper.map_type(&tcx, &ty);
            let tensor_type = create_triton_ranked_tensor(ty);

            create_ub_poison(
                self.module.context(),
                Location::unknown(self.module.context()),
                tensor_type,
            )
        } else {
            todo!(
                "name: {:?}, Adt: {:?}, adt_def: {:?}, variant_idx: {:?}, raw_list: {:?}, user_type_annotation_index: {:?}, field_idx: {:?}",
                name,
                def_id,
                adt_def,
                variant_idx,
                raw_list,
                user_type_annotation_index,
                field_idx
            )
        }
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
        terminator: &Terminator<'tcx>,
        mlir_block: &BlockRef<'a, 'blk>,
        ssa_values: &mut HashMap<rustc_middle::mir::Local, melior::ir::Value<'a, 'a>>,
    ) -> Result<(), MlirError> {
        match &terminator.kind {
            rustc_middle::mir::TerminatorKind::Return => {
                self.codegen_return(terminator, mlir_block, ssa_values)
            }
            _ => todo!("Not yet implemented - terminator: {:?}", terminator.kind),
        }
    }

    fn codegen_return<'tcx, 'blk>(
        &mut self,
        _terminator: &Terminator<'tcx>,
        mlir_block: &BlockRef<'a, 'blk>,
        ssa_values: &mut HashMap<rustc_middle::mir::Local, melior::ir::Value<'a, 'a>>,
    ) -> Result<(), MlirError> {
        println!("[DEBUG] TritonCodegen::codegen_return: ssa_values: {:?}", ssa_values);
        let value = ssa_values.get(&Local::ZERO).copied();
        let return_op = create_tt_return(
            self.module.context(),
            Location::unknown(self.module.context()),
            value.as_slice(),
        );
        mlir_block.append_operation(return_op.into());
        Ok(())
    }

    fn codegen_cast<'tcx, 'blk>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        cast_kind: &CastKind,
        operand: &Operand<'tcx>,
        ty: &Ty<'tcx>,
        mlir_block: &BlockRef<'a, 'blk>,
    ) -> Operation<'a> {
        match cast_kind {
            CastKind::PointerExposeProvenance => todo!("PointerExposeProvenance"),
            CastKind::PointerWithExposedProvenance => {
                self.codegen_pointer_with_exposed_provenance(tcx, instance, operand, ty, mlir_block)
            }
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
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        operand: &Operand<'tcx>,
        ty: &Ty<'tcx>,
        mlir_block: &BlockRef<'a, 'blk>,
    ) -> Operation<'a> {
        // Resolve function-level type parameters (e.g. D) using this instance's generic args,
        // then normalize (e.g. associated types).
        let typing_env = TypingEnv::fully_monomorphized();
        let normalized_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            typing_env,
            EarlyBinder::bind(*ty),
        );

        println!(
            "[DEBUG] TritonCodegen::codegen_pointer_with_exposed_provenance: provenance: {:?} ty: {:?} normalized: {:?}",
            operand, ty, normalized_ty
        );

        match operand {
            Operand::Copy(place) => todo!("Copy place: {:?}", place),
            Operand::Move(place) => todo!("Move place: {:?}", place),
            Operand::Constant(const_operand) => {
                self.codegen_constant_cast(tcx, const_operand, normalized_ty, mlir_block)
            }
        }
    }

    fn codegen_constant_cast<'tcx, 'blk>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        const_operand: &ConstOperand<'tcx>,
        normalized_ty: Ty<'tcx>,
        mlir_block: &BlockRef<'a, 'blk>,
    ) -> Operation<'a> {
        match const_operand.const_ {
            Const::Ty(ty, const_val) => todo!("Ty ty: {:?}, const_val: {:?}", ty, const_val),
            Const::Unevaluated(unevaluated_const, ty) => {
                todo!(
                    "Unevaluated unevaluated_const: ty: {:?} | {:?} | {:?} | ty: {:?}",
                    const_operand.ty(),
                    normalized_ty,
                    unevaluated_const,
                    ty
                )
            }
            Const::Val(const_val, ty) => {
                let const_op = self.codegen_const_value(const_val, ty);
                match normalized_ty.kind() {
                    TyKind::RawPtr(_, _) => {
                        let result = const_op.result().unwrap();
                        let result_ty = result.r#type();
                        debug_assert!(
                            result_ty.is_integer(),
                            "Triton supports only integer pointer casts"
                        );
                        let ptr_ty = self.type_mapper.map_type(&tcx, &normalized_ty);
                        let cast_op = create_tt_int_to_ptr_cast(
                            self.module.context(),
                            Location::unknown(self.module.context()),
                            result.into(),
                            ptr_ty,
                        )
                        .into();

                        mlir_block.append_operation(const_op.into());
                        cast_op
                    }
                    _ => todo!("Constant cast normalized_ty: {:?}", normalized_ty),
                }
            }
        }
    }

    fn codegen_const_value<'tcx>(
        &mut self,
        const_val: ConstValue,
        ty: Ty<'tcx>,
    ) -> ConstantOperation<'a> {
        match const_val {
            ConstValue::Scalar(scalar) => self.codegen_scalar_const_value(scalar, ty),
            ConstValue::ZeroSized => todo!("ZeroSized"),
            ConstValue::Slice { alloc_id, meta } => {
                todo!("Slice alloc_id: {:?}, meta: {:?}", alloc_id, meta)
            }
            ConstValue::Indirect { alloc_id, offset } => {
                todo!("Indirect alloc_id: {:?}, offset: {:?}", alloc_id, offset)
            }
        }
    }

    fn codegen_scalar_const_value<'tcx>(
        &mut self,
        scalar: Scalar,
        ty: Ty<'tcx>,
    ) -> ConstantOperation<'a> {
        match scalar {
            Scalar::Int(int) => match ty.kind() {
                TyKind::Uint(UintTy::Usize) => {
                    let value = int.to_i64();
                    create_constant(
                        self.module.context(),
                        Location::unknown(self.module.context()),
                        Int::I64(value as i64),
                    )
                }
                _ => todo!("Scalar::Int ty: {:?}", ty),
            },
            Scalar::Ptr(ptr, size) => todo!("Ptr ptr: {:?}, size: {:?}", ptr, size),
        }
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
