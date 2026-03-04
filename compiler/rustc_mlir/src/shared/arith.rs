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

use melior::Context;
use melior::dialect::ods::arith::{ConstantOperation, MulIOperation};
use melior::ir::r#type::IntegerType;
use melior::ir::{TypeLike, Value, ValueLike};

use crate::errors::Error;

pub enum Int {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    I128(i128),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
}

use rustc_middle::ty::{ScalarInt, Ty, TyCtxt, TyKind};

/// Creates a MLIR arith.constant operation for the given ScalarInt value and Rust type.
///
/// # Arguments
/// - `context`: MLIR context
/// - `location`: MLIR location for the operation
/// - `ty`: Rust type (`Ty<'tcx>`) describing the integer type
/// - `scalar`: The constant value as ScalarInt
///
/// # Returns
/// On success, returns a `ConstantOperation` representing the constant integer.
/// On failure (type not supported or kind mismatch), returns an Error.
pub fn create_constant_from_scalar<'ctx, 'tcx>(
    context: &'ctx Context,
    location: melior::ir::Location<'ctx>,
    ty: Ty<'tcx>,
    scalar: ScalarInt,
) -> Result<ConstantOperation<'ctx>, Error> {
    // AXM: TODO: fix this with the correct integer type for unsigned integers, we need to
    // to use the signless variant of the integer type

    // Only support integer types for now
    match ty.kind() {
        TyKind::Int(int_ty) => {
            let value = match int_ty {
                rustc_ast::IntTy::I8 => Int::I8(scalar.to_i8()),
                rustc_ast::IntTy::I16 => Int::I16(scalar.to_i16()),
                rustc_ast::IntTy::I32 => Int::I32(scalar.to_i32()),
                rustc_ast::IntTy::I64 => Int::I64(scalar.to_i64()),
                rustc_ast::IntTy::I128 => Int::I128(scalar.to_i128()),
                rustc_ast::IntTy::Isize => {
                    // Not supported or device-dependent, so error here
                    return Err(Error::InvalidType {
                        msg: "isize is device-dependent and not supported".to_string(),
                    });
                }
            };
            create_constant(context, location, value)
        }
        TyKind::Uint(uint_ty) => {
            let value = match uint_ty {
                rustc_ast::UintTy::U8 => Int::U8(scalar.to_u8()),
                rustc_ast::UintTy::U16 => Int::U16(scalar.to_u16()),
                rustc_ast::UintTy::U32 => Int::U32(scalar.to_u32()),
                rustc_ast::UintTy::U64 => Int::U64(scalar.to_u64()),
                rustc_ast::UintTy::U128 => Int::U128(scalar.to_u128()),
                rustc_ast::UintTy::Usize => {
                    return Err(Error::InvalidType {
                        msg: "usize is device-dependent and not supported".to_string(),
                    });
                }
            };
            create_constant(context, location, value)
        }
        _ => Err(Error::InvalidType { msg: format!("Unsupported type for constant: {:?}", ty) }),
    }
}

pub fn create_constant<'ctx>(
    context: &'ctx Context,
    location: melior::ir::Location<'ctx>,
    value: Int,
) -> Result<ConstantOperation<'ctx>, Error> {
    // AXM: TODO: fix this with the correct integer type for unsigned integers
    let attr_source = match value {
        Int::I8(value) => (format!("{} : i8", value), IntegerType::new(context, 8).into()),
        Int::I16(value) => (format!("{} : i16", value), IntegerType::new(context, 16).into()),
        Int::I32(value) => (format!("{} : i32", value), IntegerType::new(context, 32).into()),
        Int::I64(value) => (format!("{} : i64", value), IntegerType::new(context, 64).into()),
        Int::I128(value) => (format!("{} : i128", value), IntegerType::new(context, 128).into()),
        Int::U8(value) => (format!("{} : i8", value), IntegerType::new(context, 8).into()),
        Int::U16(value) => (format!("{} : i16", value), IntegerType::new(context, 16).into()),
        Int::U32(value) => (format!("{} : i32", value), IntegerType::new(context, 32).into()),
        Int::U64(value) => (format!("{} : i64", value), IntegerType::new(context, 64).into()),
        Int::U128(value) => (format!("{} : i128", value), IntegerType::new(context, 128).into()),
    };

    // Create a zero constant of type i64 using the arith dialect
    let attr = melior::ir::Attribute::parse(context, &attr_source.0)
        .unwrap_or_else(|| panic!("failed to parse attribute: {}", attr_source.0));

    Ok(melior::dialect::ods::arith::ConstantOperation::builder(context, location)
        .value(attr)
        .result(attr_source.1)
        .build())
}

pub fn create_muli<'ctx>(
    context: &'ctx Context,
    location: melior::ir::Location<'ctx>,
    lhs: Value<'ctx, 'ctx>,
    rhs: Value<'ctx, 'ctx>,
) -> Result<MulIOperation<'ctx>, Error> {
    let lhs_ty = lhs.r#type();
    let rhs_ty = rhs.r#type();

    if lhs_ty != rhs_ty {
        return Err(Error::IncompatibleTypes { lhs: lhs_ty.to_string(), rhs: rhs_ty.to_string() });
    }

    if !lhs_ty.is_integer() {
        return Err(Error::InvalidType { msg: lhs_ty.to_string() });
    }

    Ok(melior::dialect::ods::arith::MulIOperation::builder(context, location)
        .lhs(lhs)
        .rhs(rhs)
        .build())
}

#[cfg(test)]
mod tests {
    use melior::ir::operation::OperationLike;
    use melior::ir::{BlockLike, Location, Module, Operation};

    use super::*;
    use crate::test::create_test_context;

    #[test]
    fn test_create_constant() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        let constant_op = create_constant(&context, location, Int::I64(0)).unwrap();

        let expected = "%c0_i64 = arith.constant 0 : i64\n";
        let output = constant_op.as_operation().to_string();
        assert_eq!(expected, output);
    }

    #[test]
    fn test_create_muli() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let module = Module::new(location);

        // Create two i32 constants
        let lhs: Operation = create_constant(&context, location, Int::I32(4)).unwrap().into();
        let rhs: Operation = create_constant(&context, location, Int::I32(5)).unwrap().into();

        // Get their values
        let lhs_value = lhs.result(0).unwrap().into();
        let rhs_value = rhs.result(0).unwrap().into();

        // Generate arith.muli operation
        let muli = create_muli(&context, location, lhs_value, rhs_value).unwrap().into();

        module.body().append_operation(lhs);
        module.body().append_operation(rhs);
        module.body().append_operation(muli);

        let expected = "module {\n  %c4_i32 = arith.constant 4 : i32\n  %c5_i32 = arith.constant 5 : i32\n  %0 = arith.muli %c4_i32, %c5_i32 : i32\n}\n";
        let output = module.as_operation().to_string();
        assert_eq!(expected, output);
    }
}
