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
use mlir_sys::MlirContext;

use crate::ffi::mlirLoadTritonDialect;

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

#[cfg(test)]
mod tests {
    use melior::ir::attribute::{StringAttribute, TypeAttribute};
    use melior::ir::r#type::FunctionType;
    use melior::ir::{Location, Module, Type};

    use super::tt;
    use crate::test::create_test_context;
    use crate::triton::load_triton_dialect;

    #[test]
    fn test_tt_func() {
        let context = create_test_context();
        load_triton_dialect(&context);

        println!("AXM: Loaded dialects: {:?}", context.loaded_dialect_count());
        println!("AXM: Registered dialects: {:?}", context.registered_dialect_count());
        let tt_dialect = context.get_or_load_dialect("tt");
        println!("AXM: TT dialect: {:?}", tt_dialect);

        let location = Location::unknown(&context);
        let module = Module::new(location);
        let _body = module.body();

        let ptr_f32_type = Type::parse(&context, "tt.ptr<f32>").expect("Failed to parse ptr<f32>");
        let inputs = vec![ptr_f32_type];
        let function_type = TypeAttribute::new(FunctionType::new(&context, &inputs, &[]).into());

        let builder = tt::FuncOperation::builder(&context, location)
            .sym_name(StringAttribute::new(&context, "test_tt_func"));
        builder.function_type(function_type);
    }
}
