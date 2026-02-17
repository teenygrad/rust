/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

use melior::Context;
use melior::ir::{Type, TypeLike};

use crate::ffi::{mlirLoadTritonDialect, mlirTritonPointerType};

melior_macro::dialect! {
    name: "tt",
    files: [
        "triton/Dialect/Triton/IR/TritonDialect.td",
        "triton/Dialect/Triton/IR/TritonOps.td",
        "triton/Dialect/Triton/IR/TritonTypes.td"
    ],
    include_directories: ["TRITON_INCLUDE_DIRECTORY"],
}

pub fn load_triton_dialect(context: &Context) {
    unsafe {
        mlirLoadTritonDialect(context.to_raw());
    }
}
pub fn triton_pointer_type<'a>(pointee: &Type<'a>) -> Type<'a> {
    unsafe { Type::from_raw(mlirTritonPointerType(pointee.to_raw(), 1)) }
}

#[cfg(test)]
mod tests {
    use melior::ir::attribute::{StringAttribute, TypeAttribute};
    use melior::ir::r#type::FunctionType;
    use melior::ir::{BlockLike, Location, Module, Region, Type};

    use super::tt::*;
    use crate::test::create_test_context;
    use crate::triton::{load_triton_dialect, triton_pointer_type};

    #[test]
    fn test_tt_func_op() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);
        let _body = module.body();

        let f32_type = Type::float32(&context);
        let ptr_f32_type = triton_pointer_type(&f32_type);

        let inputs = vec![ptr_f32_type];
        let function_type =
            TypeAttribute::new(FunctionType::new(&context, &inputs, &[f32_type]).into());

        let body_region = Region::new();
        let builder = FuncOperation::builder(&context, location)
            .sym_name(StringAttribute::new(&context, "test_tt_func"))
            .function_type(function_type)
            .sym_visibility(StringAttribute::new(&context, "private"))
            .body(body_region);
        let func_op = builder.build();

        module.body().append_operation(func_op.into());

        let expected = "module {\n  tt.func private @test_tt_func(!tt.ptr<f32>) -> f32\n}\n";
        assert_eq!(expected, module.as_operation().to_string());
    }
}
