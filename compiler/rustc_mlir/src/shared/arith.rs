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
use melior::dialect::ods::arith::ConstantOperation;
use melior::ir::r#type::IntegerType;

pub enum Int {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
}

pub fn create_constant<'a>(
    context: &'a Context,
    location: melior::ir::Location<'a>,
    value: Int,
) -> ConstantOperation<'a> {
    let attr_source = match value {
        Int::I8(value) => (format!("{} : i8", value), IntegerType::new(context, 8).into()),
        Int::I16(value) => (format!("{} : i16", value), IntegerType::new(context, 16).into()),
        Int::I32(value) => (format!("{} : i32", value), IntegerType::new(context, 32).into()),
        Int::I64(value) => (format!("{} : i64", value), IntegerType::new(context, 64).into()),
    };

    // Create a zero constant of type i64 using the arith dialect
    let attr =
        melior::ir::Attribute::parse(context, &attr_source.0).expect("failed to parse attribute");

    melior::dialect::ods::arith::ConstantOperation::builder(context, location)
        .value(attr)
        .result(attr_source.1)
        .build()
}

#[cfg(test)]
mod tests {
    use melior::ir::Location;

    use super::*;
    use crate::test::create_test_context;

    #[test]
    fn test_create_constant() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        let constant_op = create_constant(&context, location, Int::I64(0));

        let expected = "%c0_i64 = arith.constant 0 : i64\n";
        let output = constant_op.as_operation().to_string();
        assert_eq!(expected, output);
    }
}
