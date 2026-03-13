
#include "llvm/IR/Verifier.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "MLIRWrapper.h"

extern "C" MLIRContextRef MLIRRustContextCreate() {
  return wrap(new MLIRContext());
}

extern "C" MLIRRustResult MLIRRustInitTriton(MLIRContextRef ctx) {
  MLIRContext *context = unwrap(ctx);

  llvm::errs() << "Initializing Triton dialect\n";

  DialectRegistry registry;
  registry.insert<BuiltinDialect, DLTIDialect, LLVM::LLVMDialect,
                  func::FuncDialect>();
  registry.insert<mlir::triton::TritonDialect>();
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();

  return MLIRRustResult::Success;
}

extern "C" OpBuilderRef MLIRRustModuleBuilderCreate(MLIRContextRef ctx) {
  return wrap(new OpBuilder(unwrap(ctx)));
}

extern "C" ModuleRef MLIRRustModuleCreate(OpBuilderRef builder_ref) {
  auto builder = unwrap(builder_ref);
  auto module = new mlir::OwningOpRef<ModuleOp>(
      ModuleOp::create(builder->getUnknownLoc()));

  return wrap(module);
}
