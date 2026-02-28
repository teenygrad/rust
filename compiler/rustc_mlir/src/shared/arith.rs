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
}

pub fn create_constant<'ctx>(
    context: &'ctx Context,
    location: melior::ir::Location<'ctx>,
    value: Int,
) -> Result<ConstantOperation<'ctx>, Error> {
    let attr_source = match value {
        Int::I8(value) => (format!("{} : i8", value), IntegerType::new(context, 8).into()),
        Int::I16(value) => (format!("{} : i16", value), IntegerType::new(context, 16).into()),
        Int::I32(value) => (format!("{} : i32", value), IntegerType::new(context, 32).into()),
        Int::I64(value) => (format!("{} : i64", value), IntegerType::new(context, 64).into()),
    };

    // Create a zero constant of type i64 using the arith dialect
    let attr =
        melior::ir::Attribute::parse(context, &attr_source.0).expect("failed to parse attribute");

    Ok(melior::dialect::ods::arith::ConstantOperation::builder(context, location)
        .value(attr)
        .result(attr_source.1)
        .build())
}

pub fn create_muli<'ctx, 'a>(
    context: &'a Context,
    location: melior::ir::Location<'a>,
    lhs: Value<'ctx, 'a>,
    rhs: Value<'ctx, 'a>,
) -> Result<MulIOperation<'a>, Error>
where
    'ctx: 'a,
{
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
