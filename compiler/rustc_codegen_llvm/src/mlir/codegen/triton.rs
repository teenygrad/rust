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

use rustc_middle::mir::mono::MonoItem;
use rustc_middle::ty::TyCtxt;

use crate::mlir::MlirModule;
use crate::mlir::codegen::Codegen;
use crate::mlir::errors::MlirError;

pub(crate) struct TritonCodegen<'a> {
    module: &'a mut MlirModule,
}

impl<'a> TritonCodegen<'a> {
    pub(crate) fn new(module: &'a mut MlirModule) -> Self {
        Self { module }
    }
}

impl<'a> Codegen for TritonCodegen<'a> {
    fn codegen<'tcx>(&mut self, tcx: TyCtxt<'tcx>, item: &MonoItem<'tcx>) -> Result<(), MlirError> {
        match item {
            MonoItem::Fn(instance) => {
                // Get the mangled function name
                let mangled_name = tcx.symbol_name(*instance);
                eprintln!("[DEBUG] Function mangled name: {}", mangled_name);
                // TODO: Implement Triton codegen
                todo!()
            }
            MonoItem::Static(_def_id) => {
                // TODO: Implement Triton codegen for statics
                todo!()
            }
            MonoItem::GlobalAsm(_item_id) => {
                // TODO: Implement Triton codegen for global asm
                todo!()
            }
        }
    }
}
