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
use melior::ir::Location;
use melior::ir::r#type::IntegerType;

use crate::errors::Error;
use crate::shared::builtin::create_tensor_type;
use crate::triton::attr_i32;
use crate::triton::tt::MakeRangeOperation;

pub fn create_arange<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    start: i32,
    end: i32,
) -> Result<MakeRangeOperation<'ctx>, Error> {
    let start_attr = attr_i32(context, start);
    let end_attr = attr_i32(context, end);
    let element_type = IntegerType::new(context, 32).into();
    let dimensions = &[(end - start) as i64];

    let result = create_tensor_type(dimensions, element_type).into();
    Ok(MakeRangeOperation::builder(context, location)
        .start(start_attr)
        .end(end_attr)
        .result(result)
        .build())
}

#[cfg(test)]
mod tests {
    use melior::Context;
    use melior::ir::Location;

    use super::*;

    #[test]
    fn test_create_arange() {
        let context = Context::new();
        let location = Location::unknown(&context);
        let start = 0;
        let end = 5;

        let arange_op = create_arange(&context, location, start, end);
        assert!(arange_op.is_ok());
        let op = arange_op.unwrap();

        let output = op.as_operation().to_string();
        let expected =
            "%0 = \"tt.make_range\"() {end = 5 : i32, start = 0 : i32} : () -> tensor<5xi32>\n";
        assert_eq!(expected, output);
    }
}
