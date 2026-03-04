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
use melior::ir::{BlockLike, BlockRef, Location, Operation, TypeLike, Value, ValueLike};
use rustc_middle::mir::{Body, Place};
use rustc_middle::ty::{Instance, TyCtxt};
use rustc_mlir::shared::arith::create_muli;

use crate::mlir::codegen::triton::{SsaValues, TritonCodegen};
use crate::mlir::errors::MlirError;

impl<'a> TritonCodegen<'a> {
    pub fn codegen_mul<'tcx>(
        &self,
        lhs: Value<'a, 'a>,
        rhs: Value<'a, 'a>,
        mlir_block: &BlockRef,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let lhs_ty = lhs.r#type();
        let rhs_ty = rhs.r#type();

        if lhs_ty.is_integer() {
            let mul_op: Operation = create_muli(
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
            todo!("TritonCodegen::codegen_mul: {:?} {:?}", lhs_ty, rhs_ty);
        }
    }
}
