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
use melior::dialect::ods::arith::{
    AddIOperation, ConstantOperation, ExtSIOperation, MulIOperation,
};
use melior::ir::r#type::IntegerType;
use melior::ir::{Attribute, Location, Type, TypeLike, Value, ValueLike};

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

use rustc_middle::ty::{ScalarInt, Ty, TyKind};

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
pub fn create_scalar_attr<'ctx, 'tcx>(
    context: &'ctx Context,
    ty: Ty<'tcx>,
    scalar: ScalarInt,
) -> Result<(Attribute<'ctx>, Type<'ctx>), Error> {
    // AXM: TODO: fix this with the correct integer type for unsigned integers, we need to
    // to use the signless variant of the integer type

    // Only support integer types for now
    let attr_source = match ty.kind() {
        TyKind::Int(int_ty) => match int_ty {
            rustc_ast::IntTy::I8 => {
                (format!("{} : i8", scalar.to_i8()), IntegerType::new(context, 8).into())
            }
            rustc_ast::IntTy::I16 => {
                (format!("{} : i16", scalar.to_i16()), IntegerType::new(context, 16).into())
            }
            rustc_ast::IntTy::I32 => {
                (format!("{} : i32", scalar.to_i32()), IntegerType::new(context, 32).into())
            }
            rustc_ast::IntTy::I64 => {
                (format!("{} : i64", scalar.to_i64()), IntegerType::new(context, 64).into())
            }
            rustc_ast::IntTy::I128 => {
                (format!("{} : i128", scalar.to_i128()), IntegerType::new(context, 128).into())
            }
            rustc_ast::IntTy::Isize => {
                return Err(Error::InvalidType {
                    msg: "isize is device-dependent and not supported".to_string(),
                });
            }
        },
        TyKind::Uint(uint_ty) => match uint_ty {
            rustc_ast::UintTy::U8 => {
                (format!("{} : i8", scalar.to_u8()), IntegerType::new(context, 8).into())
            }
            rustc_ast::UintTy::U16 => {
                (format!("{} : i16", scalar.to_u16()), IntegerType::new(context, 16).into())
            }
            rustc_ast::UintTy::U32 => {
                (format!("{} : i32", scalar.to_u32()), IntegerType::new(context, 32).into())
            }
            rustc_ast::UintTy::U64 => {
                (format!("{} : i64", scalar.to_u64()), IntegerType::new(context, 64).into())
            }
            rustc_ast::UintTy::U128 => {
                (format!("{} : i128", scalar.to_u128()), IntegerType::new(context, 128).into())
            }
            rustc_ast::UintTy::Usize => {
                return Err(Error::InvalidType {
                    msg: "usize is device-dependent and not supported".to_string(),
                });
            }
        },
        _ => {
            return Err(Error::InvalidType {
                msg: format!("Unsupported type for constant: {:?}", ty),
            });
        }
    };

    Ok((
        Attribute::parse(context, &attr_source.0)
            .unwrap_or_else(|| panic!("failed to parse attribute: {}", attr_source.0)),
        attr_source.1,
    ))
}

pub fn create_constant<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    attr: Attribute<'ctx>,
    result_ty: Type<'ctx>,
) -> Result<ConstantOperation<'ctx>, Error> {
    Ok(ConstantOperation::builder(context, location).value(attr).result(result_ty).build())
}

pub fn create_muli<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
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

    Ok(MulIOperation::builder(context, location).lhs(lhs).rhs(rhs).build())
}

pub fn create_addi<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    lhs: Value<'ctx, 'ctx>,
    rhs: Value<'ctx, 'ctx>,
) -> Result<AddIOperation<'ctx>, Error> {
    Ok(AddIOperation::builder(context, location).lhs(lhs).rhs(rhs).build())
}

pub fn create_extsi<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    src: Value<'ctx, 'ctx>,
    result_ty: Type<'ctx>,
) -> Result<ExtSIOperation<'ctx>, Error> {
    Ok(ExtSIOperation::builder(context, location).r#in(src).out(result_ty).build())
}

#[cfg(test)]
mod tests {

    use melior::ir::operation::OperationLike;
    use melior::ir::{BlockLike, Location, Module, Operation};
    use rustc_middle::ty::TyCtxt;

    use super::*;
    use crate::test::create_test_context;

    #[test]
    fn test_create_constant() {
        todo!();
        // let context = create_test_context();
        // let location = Location::unknown(&context);

        // let constant_op = create_constant(&context, location, Int::I64(0)).unwrap();

        // let expected = "%c0_i64 = arith.constant 0 : i64\n";
        // let output = constant_op.as_operation().to_string();
        // assert_eq!(expected, output);
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

    #[test]
    fn test_create_addi() {
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
        let addi = create_addi(&context, location, lhs_value, rhs_value).unwrap().into();

        module.body().append_operation(lhs);
        module.body().append_operation(rhs);
        module.body().append_operation(addi);

        let expected = "module {\n  %c4_i32 = arith.constant 4 : i32\n  %c5_i32 = arith.constant 5 : i32\n  %0 = arith.muli %c4_i32, %c5_i32 : i32\n}\n";
        let output = module.as_operation().to_string();
        assert_eq!(expected, output);
    }
}
