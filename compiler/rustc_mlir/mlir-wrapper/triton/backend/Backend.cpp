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

#include <iostream>

#include "Backend.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace triton {

Backend::Backend(std::string target) : m_target(target) {
  // nop
}

Backend::~Backend() {
  // nop
}

void Backend::printIR(std::string stage, ModuleOp module) {
  llvm::outs() << "--------------------------------\n";
  llvm::outs() << "Stage: " << stage << "\n";
  module.print(llvm::outs());
  llvm::outs() << "\n--------------------------------\n";
}

std::optional<Error> Backend::addPass(PassManager& pm, MlirPass pass) {
  auto pass_fn = m_pass_fns.find(pass);
  if (pass_fn == m_pass_fns.end()) {
    m_last_error = std::make_optional(Error::InvalidPass);
    m_last_error_string = "Invalid triton pass";
    return m_last_error;
  }

  pm.addPass(pass_fn->second());
  return std::nullopt;
}

std::optional<Error> Backend::addPass(PassManager& pm, MlirPass pass, int arg0) {
  m_last_error = std::nullopt;
  m_last_error_string = "";

  switch (pass) {
    case MlirPass::ttgpuir_assign_latencies:
      pm.addPass(createTritonGPUAssignLatencies({arg0}));
      break;

    case MlirPass::ttgpuir_warp_specialize:
      pm.addPass(createTritonGPUAutomaticWarpSpecialization({arg0}));
      break;

    default:
      m_last_error = std::make_optional(Error::InvalidPass);
      m_last_error_string = "Invalid triton pass";
      break;
  }

  return m_last_error;
}

std::optional<Error> Backend::addPass(PassManager& pm, MlirPass pass, bool arg0) {
  if (pass != MlirPass::ttgpuir_optimize_dot_operands) {
    m_last_error = std::make_optional(Error::InvalidPass);
    m_last_error_string = "Invalid triton pass";
    return m_last_error;
  }

  pm.addPass(createTritonGPUOptimizeDotOperands({arg0}));
  return std::nullopt;
}

std::optional<Error> Backend::addPass(PassManager& pm, MlirPass pass, int arg0, bool arg1) {
  if (pass != MlirPass::ttgpuir_pipeline) {
    m_last_error = std::make_optional(Error::InvalidPass);
    m_last_error_string = "Invalid triton pass";
    return m_last_error;
  }

  pm.addPass(createTritonGPUPipeline({arg0, arg1}));
  return std::nullopt;
}

std::optional<Error> Backend::addPass(PassManager& pm, MlirPass pass, const std::string &arg0,
  int arg1, int arg2, int arg3) {
  if (pass != MlirPass::ttir_convert_to_ttgpuir) {
    m_last_error = std::make_optional(Error::InvalidPass);
    m_last_error_string = "Invalid triton pass";
    return m_last_error;
  }

  pm.addPass(createConvertTritonToTritonGPU({arg0, arg1, arg2, arg3}));
  return std::nullopt;
}

}
}
