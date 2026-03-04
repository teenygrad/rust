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
    pub fn codegen_program_id<'tcx>(
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

        debug_assert!(args.len() == 1, "TritonCodegen::codegen_program_id: args length must be 1");

        let axis = match &args[0].node {
            Operand::Constant(c) => {
                // We expect the constant to have a value that tells us the discriminant/variant
                match c.const_ {
                    Const::Val(ConstValue::Scalar(Scalar::Int(scalar_int)), _) => {
                        <ProgramAxis as From<i32>>::from(
                            scalar_int.to_bits(scalar_int.size()) as i32
                        )
                    }
                    _ => {
                        todo!(
                            "TritonCodegen::codegen_program_id: axis does not have scalar discriminant value"
                        );
                    }
                }
            }
            _ => todo!("TritonCodegen::codegen_program_id: axis must be a constant"),
        };

        let program_id_op: Operation<'a> = create_tt_program_id(
            self.module.context(),
            Location::unknown(self.module.context()),
            axis,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();

        let result = program_id_op.result(0).expect("Program ID operation result not found");
        mlir_block.append_operation(program_id_op);
        Ok(result.into())
    }
}
