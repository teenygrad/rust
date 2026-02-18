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
    use melior::dialect::ods::arith;
    use melior::ir::attribute::{
        ArrayAttribute, Attribute, BoolAttribute, StringAttribute, TypeAttribute,
    };
    use melior::ir::operation::OperationMutLike;
    use melior::ir::r#type::FunctionType;
    use melior::ir::{Block, BlockLike, Location, Module, Operation, Region, RegionLike, Type};

    use super::tt::*;
    use crate::test::create_test_context;
    use crate::triton::{load_triton_dialect, triton_pointer_type};

    #[test]
    fn test_tt_func_op_with_attributes() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = Type::float32(&context);
        let ptr_f32_type = triton_pointer_type(&f32_type);

        // Function signature: (!tt.ptr<f32>, !tt.ptr<f32>) -> f32
        let inputs = vec![ptr_f32_type, ptr_f32_type];
        let results = vec![f32_type];
        let function_type =
            TypeAttribute::new(FunctionType::new(&context, &inputs, &results).into());

        // Argument attributes: one dictionary per argument
        // arg0: {tt.divisibility = 16 : i32, tt.contiguity = 1 : i32}
        // arg1: {tt.divisibility = 16 : i32}
        let arg0_attrs =
            Attribute::parse(&context, "{tt.divisibility = 16 : i32, tt.contiguity = 1 : i32}")
                .expect("valid arg0 attrs");
        let arg1_attrs =
            Attribute::parse(&context, "{tt.divisibility = 16 : i32}").expect("valid arg1 attrs");
        let arg_attrs = ArrayAttribute::new(&context, &[arg0_attrs, arg1_attrs]);

        // Result attributes: one dictionary per result
        // result0: {tt.constancy = 1 : i32}
        let res0_attrs =
            Attribute::parse(&context, "{tt.constancy = 1 : i32}").expect("valid res0 attrs");
        let res_attrs = ArrayAttribute::new(&context, &[res0_attrs]);

        let body_region = Region::new();
        let func_op = FuncOperation::builder(&context, location)
            .sym_name(StringAttribute::new(&context, "test_tt_func_attrs"))
            .function_type(function_type)
            .sym_visibility(StringAttribute::new(&context, "public"))
            .arg_attrs(arg_attrs)
            .res_attrs(res_attrs)
            .body(body_region)
            .build();

        // Create a constant op returning f32 1.0
        let one_attr = Attribute::parse(&context, "1.0 : f32").expect("valid f32");
        let const_op = arith::ConstantOperation::builder(&context, location)
            .value(one_attr)
            .result(f32_type)
            .build();

        // Return from triton func with tt.return
        let return_op = ReturnOperation::builder(&context, location)
            .srcs(&[const_op.result().unwrap().into()])
            .build();

        // Insert block into function body region
        let first_block = Block::new(&[(ptr_f32_type, location), (ptr_f32_type, location)]);
        first_block.append_operation(const_op.into());
        first_block.append_operation(return_op.into());
        func_op.body().unwrap().append_block(first_block);

        // Add noinline attribute to the function operation
        let mut func_op: Operation = func_op.into();
        func_op.set_attribute("noinline", BoolAttribute::new(&context, false).into());

        module.body().append_operation(func_op);

        let output = module.as_operation().to_string();

        println!("output: {}", output);

        let expected = "module {
  tt.func public @test_tt_func_attrs(%arg0: !tt.ptr<f32> {tt.contiguity = 1 : i32, tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> (f32 {tt.constancy = 1 : i32}) attributes {noinline = false} {
    %cst = arith.constant 1.000000e+00 : f32
    tt.return %cst : f32
  }
}
";
        assert_eq!(expected, output);
    }
}
