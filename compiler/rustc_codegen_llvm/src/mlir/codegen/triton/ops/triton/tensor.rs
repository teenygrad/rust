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

use melior::ir::operation::OperationLike;
use melior::ir::{BlockLike, BlockRef, Location, Operation, Value};
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::{
    BasicBlock, Body, CallSource, Const, ConstValue, Operand, Place, UnwindAction,
};
use rustc_middle::ty::{Instance, TyCtxt};
use rustc_mlir::triton::{ProgramAxis, create_tt_program_id};
use rustc_span::Span;
use rustc_span::source_map::Spanned;

use crate::mlir::codegen::triton::{SsaValues, TritonCodegen};
use crate::mlir::errors::MlirError;

impl<'a> TritonCodegen<'a> {
    pub fn codegen_arange<'tcx>(
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
        ssa_values: &mut SsaValues<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        println!(
            "[DEBUG] TritonCodegen::codegen_program_id: func: {:?} args: {:?} destination: {:?} target: {:?} unwind: {:?} call_source: {:?} fn_span: {:?}",
            func, args, destination, target, unwind, call_source, fn_span
        );

        debug_assert!(
            args.len() == 3,
            "TritonCodegen::codegen_arange: args length must be 3: {:?}",
            args
        );

        let start = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            mlir_block,
            ssa_values,
        )?;
        let end = self.codegen_operand(
            tcx,
            instance,
            &args[1].node,
            args[1].node.ty(mir, tcx),
            mlir_block,
            ssa_values,
        )?;
        let step = self.codegen_operand(
            tcx,
            instance,
            &args[2].node,
            args[2].node.ty(mir, tcx),
            mlir_block,
            ssa_values,
        )?;

        todo!(
            "TritonCodegen::codegen_arange: start: {:?}, end: {:?}, step: {:?}",
            start,
            end,
            step
        );
    }
}
