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

use melior::ir::operation::OperationLike;
use melior::ir::r#type::TupleType;
use melior::ir::{BlockLike, BlockRef, Location, Operation, TypeLike, Value};
use rustc_middle::mir::{BasicBlock, Body, CallSource, Operand, Place, Terminator, UnwindAction};
use rustc_middle::ty::{Instance, TyCtxt, TyKind};
use rustc_mlir::shared::cf::create_cf_br;
use rustc_mlir::triton::call;
use rustc_span::Span;
use rustc_span::source_map::Spanned;

use crate::mlir::codegen::triton::{SsaValues, TritonCodegen};
use crate::mlir::errors::MlirError;

// Used inside codegen_terminator_call where 'a and 'tcx are concrete — no HRTB needed.
type LocalCallHandler<'a, 'tcx> = fn(
    &TritonCodegen<'a>,
    TyCtxt<'tcx>,
    &Instance<'tcx>,
    &Body<'tcx>,
    &Operand<'tcx>,
    &str,
    &[Spanned<Operand<'tcx>>],
    &Place<'tcx>,
    &Option<BasicBlock>,
    &UnwindAction,
    &CallSource,
    &Span,
    &BlockRef,
    &mut SsaValues<'a, 'a>,
) -> Result<Option<Value<'a, 'a>>, MlirError>;

impl<'a> TritonCodegen<'a> {
    pub fn codegen_terminator<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        terminator: &Terminator<'tcx>,
        mlir_block: &BlockRef,
        ssa_values: &mut SsaValues<'a, 'a>,
        basic_blocks: &HashMap<BasicBlock, BlockRef>,
    ) -> Result<(), MlirError> {
        println!(
            "[DEBUG] TritonCodegen::codegen_terminator: ssa_values: {:?} terminator: {:?}",
            ssa_values, terminator
        );

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
                basic_blocks,
                ssa_values,
            ),
            rustc_middle::mir::TerminatorKind::TailCall { func, args, fn_span } => {
                todo!("TailCall: {:?} {:?} {:?}", func, args, fn_span)
            }
            rustc_middle::mir::TerminatorKind::Assert { cond, expected, msg, target, unwind } => {
                todo!("Assert: {:?} {:?} {:?} {:?} {:?}", cond, expected, msg, target, unwind)
            }
            rustc_middle::mir::TerminatorKind::Yield { .. } => todo!("Yield"),
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

    fn codegen_terminator_call<'tcx>(
        &self,
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
        mlir_block: &BlockRef,
        basic_blocks: &HashMap<BasicBlock, BlockRef>,
        ssa_values: &mut SsaValues<'a, 'a>,
    ) -> Result<(), MlirError> {
        let func_name = match func {
            Operand::Constant(c) => {
                if let TyKind::FnDef(def_id, _) = c.ty().kind() {
                    tcx.def_path_str(*def_id)
                } else {
                    format!("{:?}", func)
                }
            }
            _ => format!("XX{:?}", func),
        };

        println!("[DEBUG] TritonCodegen::codegen_terminator_call: func_name: {:?}", func_name);

        let method: LocalCallHandler<'a, 'tcx> = match func_name.as_str() {
            "triton::Triton::program_id" => {
                TritonCodegen::codegen_program_id as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::arange" => TritonCodegen::codegen_arange as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::load" => TritonCodegen::codegen_load as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::store" => TritonCodegen::codegen_store as LocalCallHandler<'a, 'tcx>,
            "std::ops::Mul::mul" => TritonCodegen::codegen_mul_call as LocalCallHandler<'a, 'tcx>,
            "std::ops::Add::add" => TritonCodegen::codegen_add_call as LocalCallHandler<'a, 'tcx>,
            "triton::types::Comparison::lt" => {
                TritonCodegen::codegen_lt_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::types::AddOffsets::add_offset" => {
                TritonCodegen::codegen_add_ptr as LocalCallHandler<'a, 'tcx>
            }
            _ => TritonCodegen::codegen_call as LocalCallHandler<'a, 'tcx>,
        };

        let value = method(
            self,
            tcx,
            instance,
            mir,
            func,
            func_name.as_str(),
            args,
            destination,
            target,
            unwind,
            call_source,
            fn_span,
            mlir_block,
            ssa_values,
        )?;

        if let Some(value) = value {
            ssa_values.insert(destination.local, value);
        }
        self.codegen_goto(&target.expect("target must be Some"), mlir_block, basic_blocks)?;
        Ok(())
    }

    pub fn codegen_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        func: &Operand<'tcx>,
        func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        mlir_block: &BlockRef,
        ssa_values: &mut SsaValues<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let args = args
            .iter()
            .map(|arg| {
                self.codegen_operand(
                    tcx,
                    instance,
                    &arg.node,
                    arg.node.ty(mir, tcx),
                    mlir_block,
                    ssa_values,
                )
            })
            .collect::<Result<Vec<_>, MlirError>>()?;

        let fn_ty = func.ty(mir, tcx);
        let ret_ty = match fn_ty.kind() {
            // Function definitions (e.g., closures and fn items).
            TyKind::FnDef(def_id, substs) => {
                let sig = tcx.fn_sig(*def_id).instantiate(tcx, substs);
                sig.output().skip_binder()
            }
            // For function pointers, combine binder + header and get the output type.
            TyKind::FnPtr(binder, header) => {
                let full_sig = binder.with(*header);
                full_sig.output().skip_binder()
            }
            // Otherwise, fallback to just using the type itself.
            _ => fn_ty,
        };

        let ret_ty = self.type_mapper.map_type(self.module.context(), &tcx, &ret_ty);
        let call_op: Operation<'a> = call(
            self.module.context(),
            Location::unknown(self.module.context()),
            func_name,
            &args,
            &[ret_ty],
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();

        eprintln!("[DEBUG] AXM TritonCodegen::codegen_call: call_op: {:?}", call_op.to_string());

        let result = if call_op.result_count() > 0 {
            let result = call_op.result(0).expect("Call operation result not found");
            Some(result.into())
        } else {
            None
        };

        mlir_block.append_operation(call_op);
        Ok(result)
    }

    fn codegen_goto(
        &self,
        target: &BasicBlock,
        mlir_block: &BlockRef,
        basic_blocks: &HashMap<BasicBlock, BlockRef>,
    ) -> Result<(), MlirError> {
        let target_block = basic_blocks.get(target).unwrap();
        let br_op = create_cf_br(
            self.module.context(),
            Location::unknown(self.module.context()),
            target_block,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?;

        eprintln!(
            "[DEBUG] AXM TritonCodegen::codegen_goto: br_op: {:?}",
            br_op.as_operation().to_string()
        );
        mlir_block.append_operation(br_op.into());
        Ok(())
    }
}
