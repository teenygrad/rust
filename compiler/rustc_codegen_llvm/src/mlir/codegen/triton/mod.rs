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

use melior::Context;
use melior::dialect::ods::arith::ConstantOperation;
use melior::ir::operation::{OperationLike, OperationResult};
use melior::ir::{
    Block, BlockLike, BlockRef, Location, Operation, RegionLike, TypeLike, Value, ValueLike,
};
use melior::utility::register_all_llvm_translations;
use rustc_abi::{FieldIdx, VariantIdx};
use rustc_ast::{IntTy, UintTy};
use rustc_hir::def_id::DefId;
use rustc_index::IndexVec;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::mono::MonoItem;
use rustc_middle::mir::{
    AggregateKind, BasicBlock, BasicBlockData, BinOp, Body, CallSource, CastKind, Const,
    ConstOperand, ConstValue, Local, NonDivergingIntrinsic, Operand, Place, Rvalue, Statement,
    StatementKind, Terminator, UnwindAction,
};
use rustc_middle::ty::layout::MaybeResult;
use rustc_middle::ty::{AdtDef, EarlyBinder, GenericArg, Instance, Ty, TyCtxt, TyKind, TypingEnv};
use rustc_mlir::load_all_dialects;
use rustc_mlir::shared::arith::{Int, create_constant, create_muli};
use rustc_mlir::shared::builtin::create_tensor_type;
use rustc_mlir::shared::cf::create_cf_br;
use rustc_mlir::shared::ub::create_ub_poison;
use rustc_mlir::triton::tt::FuncOperation;
use rustc_mlir::triton::{
    create_triton_pointer_type, create_tt_func_with_divisibility, create_tt_int_to_ptr_cast,
    create_tt_return, load_triton_dialect,
};
use rustc_span::Span;
use rustc_span::source_map::Spanned;

use crate::mlir::MlirModule;
use crate::mlir::codegen::Codegen;
use crate::mlir::codegen::triton::types::TypeMapper;
use crate::mlir::errors::MlirError;

mod types;

type SsaValues<'c, 'a> = HashMap<Local, Value<'c, 'a>>;

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
        let mut ssa_values: SsaValues = HashMap::new();
        let mut basic_blocks: HashMap<BasicBlock, Block> = HashMap::new();

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
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?;

        let mir = tcx.instance_mir(instance.def);
        let location = Location::unknown(self.module.context());

        for (bb, _) in mir.basic_blocks.iter_enumerated() {
            let block = Block::new(&[]);
            if bb.index() == 0 {
                // Add function arguments as block arguments to the entry block
                for (i, ty) in arg_types.iter().enumerate() {
                    let value = block.add_argument(*ty, location);
                    ssa_values.insert(Local::from_usize(i + 1), value);
                }
            }

            basic_blocks.insert(bb, block);
        }

        for (bb, bb_data) in mir.basic_blocks.iter_enumerated() {
            self.codegen_basic_block(
                tcx,
                instance,
                &mir,
                bb,
                bb_data,
                &func_op,
                &mut ssa_values,
                &basic_blocks,
            )?;
        }

        println!("[DEBUG] TritonCodegen::codegen_function end: ssa_values: {:?}", ssa_values);
        self.module.llmod().body().append_operation(func_op.into());

        Ok(())
    }

    fn codegen_basic_block<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _bb: BasicBlock,
        bb_data: &BasicBlockData<'tcx>,
        func_op: &FuncOperation<'a>,
        ssa_values: &mut SsaValues<'a, 'a>,
        basic_blocks: &HashMap<BasicBlock, Block>,
    ) -> Result<(), MlirError> {
        // Create an empty MLIR block and append it to the function body region.
        // Block arguments will be added when argument-passing lowering is implemented.
        let mlir_block = Block::new(&[]);
        let mlir_block =
            func_op.body().expect("tt.func must have a body region").append_block(mlir_block);

        // Codegen each MIR statement in order.
        for stmt in &bb_data.statements {
            self.codegen_statement(tcx, instance, mir, stmt, &mlir_block, ssa_values)?;
        }

        // Codegen the block terminator.
        self.codegen_terminator(
            tcx,
            instance,
            mir,
            bb_data.terminator(),
            &mlir_block,
            ssa_values,
            basic_blocks,
        )?;

        Ok(())
    }

    fn codegen_statement<'tcx, 'blk>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        stmt: &Statement<'tcx>,
        mlir_block: &BlockRef<'a, 'blk>,
        ssa_values: &mut SsaValues<'a, 'a>,
    ) -> Result<(), MlirError> {
        println!("[DEBUG] TritonCodegen::codegen_statement: ssa_values: {:?}", ssa_values);
        match &stmt.kind {
            StatementKind::Assign(assign) => {
                let (place, rvalue) = assign.as_ref();
                println!(
                    "[DEBUG] TritonCodegen::codegen_statement: Assign: {:?}, {:?} {:?}",
                    stmt, place, rvalue
                );
                self.codegen_assign(tcx, instance, mir, place, rvalue, mlir_block, ssa_values)
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
        };

        println!("[DEBUG] TritonCodegen::codegen_statement: ssa_values: {:?}", ssa_values);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn codegen_assign<'tcx, 'blk>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        place: &Place<'tcx>,
        rvalue: &Rvalue<'tcx>,
        mlir_block: &BlockRef<'a, 'blk>,
        ssa_values: &mut SsaValues<'a, 'a>,
    ) -> Result<(), MlirError> {
        match rvalue {
            Rvalue::Use(operand) => {
                let ty = operand.ty(mir, tcx);
                let typing_env = TypingEnv::fully_monomorphized();
                let normalized_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    typing_env,
                    EarlyBinder::bind(ty),
                );

                let result = self.codegen_operand(
                    tcx,
                    instance,
                    operand,
                    normalized_ty,
                    mlir_block,
                    ssa_values,
                )?;
                println!(
                    "[DEBUG] TritonCodegen::codegen_assign ssa_values_insert 1: result: Place: {:?}, Result: {:?}",
                    place, result
                );
                ssa_values.insert(place.local, result);
            }
            Rvalue::Cast(cast_kind, operand, ty) => {
                println!("Cast cast_kind: {:?}, operand: {:?}, ty: {:?}", cast_kind, operand, ty);
                let result = self
                    .codegen_cast(tcx, instance, cast_kind, operand, ty, mlir_block, ssa_values)?;

                println!(
                    "[DEBUG] TritonCodegen::codegen_assign ssa_values_insert 2: result: Place: {:?}, Result: {:?}",
                    place, result
                );

                ssa_values.insert(place.local, result);
            }
            Rvalue::Aggregate(aggregate_kind, index_vec) => {
                println!(
                    "[DEBUG] TritonCodegen::codegen_assign: Aggregate: {:?}, index_vec: {:?}",
                    aggregate_kind, index_vec
                );
                println!("[DEBUG] TritonCodegen::codegen_assign: ssa_values: {:?}", ssa_values);
                let aggregate_op =
                    self.codegen_aggregate(tcx, instance, aggregate_kind, index_vec, mlir_block)?;
                let result = aggregate_op.result(0).unwrap();
                println!(
                    "[DEBUG] TritonCodegen::codegen_assign ssa_values_insert 3: result: Place: {:?}, Result: {:?}",
                    place, result
                );

                ssa_values.insert(place.local, result.into());
                mlir_block.append_operation(aggregate_op);
            }
            Rvalue::Repeat(operand, _) => todo!("Repeat: {:?}", operand),
            Rvalue::Ref(region, borrow_kind, place) => {
                todo!("Ref: {:?} {:?} {:?}", region, borrow_kind, place)
            }
            Rvalue::ThreadLocalRef(def_id) => todo!("ThreadLocalRef: {:?}", def_id),
            Rvalue::RawPtr(raw_ptr_kind, place) => todo!("RawPtr: {:?} {:?}", raw_ptr_kind, place),
            Rvalue::BinaryOp(bin_op, operands) => {
                let value = self.codegen_binary_op(
                    tcx, instance, mir, place, bin_op, operands, mlir_block, ssa_values,
                )?;
                ssa_values.insert(place.local, value);
            }
            Rvalue::NullaryOp(null_op) => todo!("NullaryOp: {:?}", null_op),
            Rvalue::UnaryOp(un_op, operand) => todo!("UnaryOp: {:?} {:?}", un_op, operand),
            Rvalue::Discriminant(place) => todo!("Discriminant: {:?}", place),
            Rvalue::ShallowInitBox(operand, ty) => todo!("ShallowInitBox: {:?} {:?}", operand, ty),
            Rvalue::CopyForDeref(place) => todo!("CopyForDeref: {:?}", place),
            Rvalue::WrapUnsafeBinder(operand, ty) => {
                todo!("WrapUnsafeBinder: {:?} {:?}", operand, ty)
            }
        }

        // todo!("[TODO] TritonCodegen::codegen_assign: {:?} {:?}", place, rvalue)
        Ok(())
    }

    fn codegen_aggregate<'tcx, 'blk>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        aggregate_kind: &AggregateKind<'tcx>,
        _index_vec: &IndexVec<FieldIdx, Operand<'tcx>>,
        _mlir_block: &BlockRef<'a, 'blk>,
    ) -> Result<Operation<'a>, MlirError> {
        match aggregate_kind {
            AggregateKind::Adt(def_id, _, raw_list, _, _) => {
                println!(
                    "[DEBUG] TritonCodegen::codegen_aggregate: Adt: {:?}, def_id: {:?}, raw_list: {:?}",
                    aggregate_kind, def_id, raw_list
                );
                let adt_def = tcx.adt_def(*def_id);
                self.codegen_adt(tcx, instance, &adt_def, raw_list.as_slice())
            }
            _ => todo!("AggregateKind: {:?}", aggregate_kind),
        }
    }

    fn codegen_adt<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        adt_def: &AdtDef<'tcx>,
        raw_list: &[GenericArg<'tcx>],
    ) -> Result<Operation<'a>, MlirError> {
        let name = format!("{:?}", adt_def);
        let map_ty = |idx: usize| {
            let ty = instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                TypingEnv::fully_monomorphized(),
                EarlyBinder::bind(raw_list[idx].expect_ty()),
            );
            self.type_mapper.map_type(&tcx, &ty)
        };

        // If the name of the ADT is tensor, then we create a poison operation.
        // This is because the tensor creation is part of the dsl dead code which
        // will be eliminated by the optimizer.
        if name == "triton::llvm::triton::tensor::Tensor" {
            let ty = map_ty(0);
            let tensor_type = create_tensor_type(&[i64::MIN], ty).into();

            Ok(create_ub_poison(
                self.module.context(),
                Location::unknown(self.module.context()),
                tensor_type,
            )
            .map_err(|e| MlirError::CreateOperation { err: e })?)
        } else if name == "triton::llvm::triton::pointer::Pointer" {
            let ty = map_ty(0);
            let pointer_type = create_triton_pointer_type(ty);

            Ok(create_ub_poison(
                self.module.context(),
                Location::unknown(self.module.context()),
                pointer_type,
            )
            .map_err(|e| MlirError::CreateOperation { err: e })?)
        } else if name == "triton::llvm::triton::num::I32" {
            debug_assert_eq!(raw_list.len(), 0, "I32 should have no arguments");
            Ok(create_constant(
                self.module.context(),
                Location::unknown(self.module.context()),
                Int::I32(0),
            )
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into())
        } else {
            todo!("Adt: {:?}", adt_def)
        }
    }

    fn codegen_binary_op<'tcx, 'blk>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        place: &Place<'tcx>,
        bin_op: &BinOp,
        operands: &(Operand<'tcx>, Operand<'tcx>),
        mlir_block: &BlockRef<'a, 'blk>,
        ssa_values: &mut SsaValues<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let (lhs_op, rhs_op) = operands;
        let lhs_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(lhs_op.ty(mir, tcx)),
        );
        let rhs_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(rhs_op.ty(mir, tcx)),
        );
        let lhs = self.codegen_operand(tcx, instance, lhs_op, lhs_ty, mlir_block, ssa_values)?;
        let rhs = self.codegen_operand(tcx, instance, rhs_op, rhs_ty, mlir_block, ssa_values)?;

        match bin_op {
            BinOp::Add => todo!(),
            BinOp::AddUnchecked => todo!(),
            BinOp::AddWithOverflow => todo!(),
            BinOp::Sub => todo!(),
            BinOp::SubUnchecked => todo!(),
            BinOp::SubWithOverflow => todo!(),
            BinOp::Mul => {
                self.codegen_mul(tcx, instance, mir, place, lhs, rhs, mlir_block, ssa_values)
            }
            BinOp::MulUnchecked => todo!(),
            BinOp::MulWithOverflow => todo!(),
            BinOp::Div => todo!(),
            BinOp::Rem => todo!(),
            BinOp::BitXor => todo!(),
            BinOp::BitAnd => todo!(),
            BinOp::BitOr => todo!(),
            BinOp::Shl => todo!(),
            BinOp::ShlUnchecked => todo!(),
            BinOp::Shr => todo!(),
            BinOp::ShrUnchecked => todo!(),
            BinOp::Eq => todo!(),
            BinOp::Lt => todo!(),
            BinOp::Le => todo!(),
            BinOp::Ne => todo!(),
            BinOp::Ge => todo!(),
            BinOp::Gt => todo!(),
            BinOp::Cmp => todo!(),
            BinOp::Offset => todo!(),
        }
    }

    fn codegen_mul<'tcx, 'blk>(
        &mut self,
        _tcx: TyCtxt<'tcx>,
        _instance: &Instance<'tcx>,
        _mir: &Body<'tcx>,
        place: &Place<'tcx>,
        lhs: Value<'a, 'a>,
        rhs: Value<'a, 'a>,
        mlir_block: &BlockRef<'a, 'blk>,
        _ssa_values: &mut SsaValues<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let lhs_ty = lhs.r#type();
        let rhs_ty = rhs.r#type();

        if lhs_ty.is_integer() {
            let mul_op: Operation<'a> = create_muli(
                self.module.context(),
                Location::unknown(self.module.context()),
                lhs,
                rhs,
            )
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
            let result = mul_op.result(0).unwrap().into();
            mlir_block.append_operation(mul_op.into());
            Ok(result)
        } else {
            todo!("TritonCodegen::codegen_mul: {:?} {:?} {:?}", lhs_ty, rhs_ty, place);
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
        println!("[DEBUG] TritonCodegen::codegen_storage_live: local: {:?}", _local);
        // NO-OP: In the context of Triton and MLIR, storage live is a no-op.
        Ok(())
    }

    fn codegen_storage_dead<'tcx, 'blk>(
        &mut self,
        _tcx: TyCtxt<'tcx>,
        _local: Local,
        _mlir_block: &BlockRef<'a, 'blk>,
    ) -> Result<(), MlirError> {
        println!("[DEBUG] TritonCodegen::codegen_storage_dead: local: {:?}", _local);
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
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        terminator: &Terminator<'tcx>,
        mlir_block: &BlockRef<'a, 'blk>,
        ssa_values: &mut SsaValues<'a, 'a>,
        basic_blocks: &HashMap<BasicBlock, Block>,
    ) -> Result<(), MlirError> {
        match &terminator.kind {
            rustc_middle::mir::TerminatorKind::Return => {
                self.codegen_return(terminator, mlir_block, ssa_values)
            }
            rustc_middle::mir::TerminatorKind::Goto { target } => {
                self.codegen_goto(target, mlir_block, basic_blocks)
            }
            rustc_middle::mir::TerminatorKind::SwitchInt { discr, targets } => {
                todo!("SwitchInt: {:?} {:?}", discr, targets)
            }
            rustc_middle::mir::TerminatorKind::UnwindResume => todo!("UnwindResume"),
            rustc_middle::mir::TerminatorKind::UnwindTerminate(unwind_terminate_reason) => {
                todo!("UnwindTerminate: {:?}", unwind_terminate_reason)
            }
            rustc_middle::mir::TerminatorKind::Unreachable => todo!("Unreachable"),
            rustc_middle::mir::TerminatorKind::Drop {
                place,
                target,
                unwind,
                replace,
                drop,
                async_fut,
            } => todo!(
                "Drop: {:?} {:?} {:?} {:?} {:?} {:?}",
                place,
                target,
                unwind,
                replace,
                drop,
                async_fut
            ),
            rustc_middle::mir::TerminatorKind::Call {
                func,
                args,
                destination,
                target,
                unwind,
                call_source,
                fn_span,
            } => self.codegen_terminator_call(
                tcx,
                instance,
                mir,
                func,
                args,
                destination,
                target,
                unwind,
                call_source,
                fn_span,
                mlir_block,
                ssa_values,
            ),
            rustc_middle::mir::TerminatorKind::TailCall { func, args, fn_span } => {
                todo!("TailCall: {:?} {:?} {:?}", func, args, fn_span)
            }
            rustc_middle::mir::TerminatorKind::Assert { cond, expected, msg, target, unwind } => {
                todo!("Assert: {:?} {:?} {:?} {:?} {:?}", cond, expected, msg, target, unwind)
            }
            rustc_middle::mir::TerminatorKind::Yield { value, resume, resume_arg, drop } => todo!(),
            rustc_middle::mir::TerminatorKind::CoroutineDrop => todo!("CoroutineDrop"),
            rustc_middle::mir::TerminatorKind::FalseEdge { real_target, imaginary_target } => {
                todo!("FalseEdge: {:?} {:?}", real_target, imaginary_target)
            }
            rustc_middle::mir::TerminatorKind::FalseUnwind { real_target, unwind } => {
                todo!("FalseUnwind: {:?} {:?}", real_target, unwind)
            }
            rustc_middle::mir::TerminatorKind::InlineAsm {
                asm_macro,
                template,
                operands,
                options,
                line_spans,
                targets,
                unwind,
            } => todo!(
                "InlineAsm: {:?} {:?} {:?} {:?} {:?} {:?} {:?}",
                asm_macro,
                template,
                operands,
                options,
                line_spans,
                targets,
                unwind
            ),
        }
    }

    fn codegen_terminator_call<'tcx, 'blk>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        func: &Operand<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        target: &Option<BasicBlock>,
        unwind: &UnwindAction,
        call_source: &CallSource,
        fn_span: &Span,
        mlir_block: &BlockRef<'a, 'blk>,
        ssa_values: &mut SsaValues<'a, 'a>,
    ) -> Result<(), MlirError> {
        let _func_name =
            self.codegen_operand(tcx, instance, func, func.ty(mir, tcx), mlir_block, ssa_values);

        todo!(
            "TritonCodegen::codegen_terminator_call: {:?} {:?} {:?} {:?} {:?} {:?} {:?}",
            func,
            args,
            destination,
            target,
            unwind,
            call_source,
            fn_span
        )
    }

    fn codegen_return<'tcx, 'blk>(
        &mut self,
        _terminator: &Terminator<'tcx>,
        mlir_block: &BlockRef<'a, 'blk>,
        ssa_values: &mut SsaValues<'a, 'a>,
    ) -> Result<(), MlirError> {
        println!("[DEBUG] TritonCodegen::codegen_return: ssa_values: {:?}", ssa_values);
        let value = ssa_values.get(&Local::ZERO).copied();
        let return_op = create_tt_return(
            self.module.context(),
            Location::unknown(self.module.context()),
            value.as_slice(),
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?;
        mlir_block.append_operation(return_op.into());
        Ok(())
    }

    fn codegen_goto<'blk>(
        &mut self,
        target: &BasicBlock,
        mlir_block: &BlockRef<'a, 'blk>,
        basic_blocks: &HashMap<BasicBlock, Block>,
    ) -> Result<(), MlirError> {
        let target_block = basic_blocks.get(target).unwrap();
        let br_op = create_cf_br(
            self.module.context(),
            Location::unknown(self.module.context()),
            target_block,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?;
        mlir_block.append_operation(br_op.into());
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
        ssa_values: &mut SsaValues<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        match cast_kind {
            CastKind::PointerWithExposedProvenance => self.codegen_pointer_with_exposed_provenance(
                tcx, instance, operand, ty, mlir_block, ssa_values,
            ),
            CastKind::PtrToPtr => {
                self.codegen_ptr_to_ptr(tcx, instance, operand, ty, mlir_block, ssa_values)
            }
            CastKind::IntToInt => {
                self.codegen_int_to_int(tcx, instance, operand, ty, mlir_block, ssa_values)
            }
            _ => todo!("CastKind: {:?}", cast_kind),
        }
    }

    fn codegen_pointer_with_exposed_provenance<'tcx, 'blk>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        operand: &Operand<'tcx>,
        ty: &Ty<'tcx>,
        mlir_block: &BlockRef<'a, 'blk>,
        ssa_values: &mut SsaValues<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
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

        self.codegen_operand(tcx, instance, operand, normalized_ty, mlir_block, ssa_values)
    }

    fn codegen_ptr_to_ptr<'tcx, 'blk>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        operand: &Operand<'tcx>,
        ty: &Ty<'tcx>,
        mlir_block: &BlockRef<'a, 'blk>,
        ssa_values: &mut SsaValues<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let typing_env = TypingEnv::fully_monomorphized();
        let normalized_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            typing_env,
            EarlyBinder::bind(*ty),
        );

        self.codegen_operand(tcx, instance, operand, normalized_ty, mlir_block, ssa_values)
    }

    fn codegen_int_to_int<'tcx, 'blk>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        operand: &Operand<'tcx>,
        ty: &Ty<'tcx>,
        mlir_block: &BlockRef<'a, 'blk>,
        ssa_values: &mut SsaValues<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let typing_env = TypingEnv::fully_monomorphized();
        let normalized_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            typing_env,
            EarlyBinder::bind(*ty),
        );
        self.codegen_operand(tcx, instance, operand, normalized_ty, mlir_block, ssa_values)
    }

    fn codegen_operand<'tcx, 'blk>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        operand: &Operand<'tcx>,
        normalized_ty: Ty<'tcx>,
        mlir_block: &BlockRef<'a, 'blk>,
        ssa_values: &mut SsaValues<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        println!("[DEBUG] TritonCodegen::codegen_operand: ssa_values: {:?}", ssa_values,);
        match operand {
            Operand::Copy(place) => {
                self.codegen_copy(tcx, instance, place, normalized_ty, ssa_values)
            }
            Operand::Move(place) => {
                // for triton move is the same as copy
                self.codegen_copy(tcx, instance, place, normalized_ty, ssa_values)
            }
            Operand::Constant(const_operand) => {
                self.codegen_constant_cast(tcx, instance, const_operand, normalized_ty, mlir_block)
            }
        }
    }

    fn codegen_copy<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        _instance: &Instance<'tcx>,
        place: &Place<'tcx>,
        normalized_ty: Ty<'tcx>,
        ssa_values: &mut SsaValues<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        println!(
            "[DEBUG] TritonCodegen::codegen_copy: ssa_values: Local: {:?}, SsaValues: {:?}",
            place.local, ssa_values
        );

        let value = ssa_values.get(&place.local).copied().expect("Value not found for local");
        let value_ty = value.r#type();
        let normalized_ty1 = self.type_mapper.map_type(&tcx, &normalized_ty);

        if value_ty != normalized_ty1 {
            todo!(
                "TritonCodegen::codegen_copy: value_ty != normalized_ty: {:?} != {:?} (ty: {:?}) instance: {:?}",
                value_ty,
                normalized_ty1,
                normalized_ty,
                _instance,
            );
        }
        Ok(value)
    }

    fn codegen_constant_cast<'tcx, 'blk>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        const_operand: &ConstOperand<'tcx>,
        normalized_ty: Ty<'tcx>,
        mlir_block: &BlockRef<'a, 'blk>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        println!("[DEBUG] TritonCodegen::codegen_constant_cast");
        match const_operand.const_ {
            Const::Val(const_val, ty) => {
                let const_op = self.codegen_const_value(tcx, instance, const_val, ty)?;
                match normalized_ty.kind() {
                    TyKind::RawPtr(_, _) => {
                        println!("[DEBUG] TritonCodegen::codegen_constant_cast: RawPtr");
                        let result = const_op.result(0).unwrap();
                        let result_ty = result.r#type();
                        debug_assert!(
                            result_ty.is_integer(),
                            "Triton supports only integer pointer casts"
                        );
                        let ptr_ty = self.type_mapper.map_type(&tcx, &normalized_ty);
                        let cast_op: Operation<'a> = create_tt_int_to_ptr_cast(
                            self.module.context(),
                            Location::unknown(self.module.context()),
                            result.into(),
                            ptr_ty,
                        )
                        .map_err(|e| MlirError::CreateOperation { err: e })?
                        .into();

                        let result = cast_op.result(0).unwrap();
                        mlir_block.append_operation(const_op);
                        mlir_block.append_operation(cast_op);
                        Ok(result.into())
                    }
                    TyKind::Adt(adt_def, args) => {
                        println!("[DEBUG] TritonCodegen::codegen_constant_cast: Adt");
                        let const_op = self.codegen_adt(tcx, instance, adt_def, args.as_slice())?;
                        let result = const_op.result(0).unwrap();
                        mlir_block.append_operation(const_op);
                        Ok(result.into())
                    }
                    _ => todo!("Constant cast normalized_ty: {:?}", normalized_ty),
                }
            }
            _ => todo!("Const: {:?}", const_operand.const_),
        }
    }

    fn codegen_const_value<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        const_val: ConstValue,
        ty: Ty<'tcx>,
    ) -> Result<Operation<'a>, MlirError> {
        match const_val {
            ConstValue::Scalar(scalar) => {
                self.codegen_scalar_const_value(tcx, instance, scalar, ty)
            }
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
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        scalar: Scalar,
        ty: Ty<'tcx>,
    ) -> Result<Operation<'a>, MlirError> {
        match scalar {
            Scalar::Int(int) => match ty.kind() {
                TyKind::Uint(UintTy::Usize) => {
                    let value = int.to_i64();
                    Ok(create_constant(
                        self.module.context(),
                        Location::unknown(self.module.context()),
                        Int::I64(value),
                    )
                    .map_err(|e| MlirError::CreateOperation { err: e })?
                    .into())
                }
                rustc_middle::infer::canonical::ir::TyKind::Adt(adt_def, args) => {
                    self.codegen_adt(tcx, instance, adt_def, args.as_slice())
                }
                _ => todo!("Scalar::Int ty: {:?} {:?}", ty.kind(), ty),
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
