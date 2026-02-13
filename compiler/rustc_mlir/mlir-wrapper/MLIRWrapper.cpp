//===-- MLIRWrapper.cpp - C bindings for MLIR types -----------------------===//
//
// Implementation of C-compatible bindings for MLIR types.
//
//===----------------------------------------------------------------------===//

#include "MLIRWrapper.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <cstring>
#include <string>

using namespace mlir;

//===----------------------------------------------------------------------===//
// Type conversion utilities using LLVM's wrap/unwrap pattern
//===----------------------------------------------------------------------===//

// Define conversion functions for each opaque type
// The pattern is: unwrap(OpaqueRef) -> C++ pointer, wrap(C++ pointer) -> OpaqueRef

#define DEFINE_SIMPLE_CONVERSION_FUNCTIONS(TYPE, REF)                          \
    static inline TYPE *unwrap(REF ref) { return reinterpret_cast<TYPE *>(ref); }   \
    static inline REF wrap(TYPE *ptr) { return reinterpret_cast<REF>(ptr); }

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(MLIRContext, MLIRContextRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(Location, MLIRLocationRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ModuleOp, MLIRModuleRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(Operation, MLIROperationRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(Block, MLIRBlockRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(Region, MLIRRegionRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(Type, MLIRTypeRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(Attribute, MLIRAttributeRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(Value, MLIRValueRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(OpBuilder, MLIROpBuilderRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(DialectRegistry, MLIRDialectRegistryRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(OperationState, MLIROperationStateRef)

// Helper to convert MLIRStringRef to llvm::StringRef
static inline llvm::StringRef unwrapStringRef(MLIRStringRef str) {
    return llvm::StringRef(str.data, str.length);
}

// Storage for Location values (they are value types, not pointers)
// We need to heap-allocate them to return stable pointers
static Location *allocLocation(Location loc) {
    return new Location(loc);
}

// Storage for Type values
static Type *allocType(Type type) {
    return new Type(type);
}

// Storage for Attribute values
static Attribute *allocAttribute(Attribute attr) {
    return new Attribute(attr);
}

// Storage for Value values
static Value *allocValue(Value val) {
    return new Value(val);
}

//===----------------------------------------------------------------------===//
// String reference operations
//===----------------------------------------------------------------------===//

MLIRStringRef mlirStringRefCreateFromCString(const char *str) {
    MLIRStringRef ref;
    ref.data = str;
    ref.length = str ? strlen(str) : 0;
    return ref;
}

MLIRStringRef mlirStringRefCreate(const char *data, size_t length) {
    MLIRStringRef ref;
    ref.data = data;
    ref.length = length;
    return ref;
}

//===----------------------------------------------------------------------===//
// Context operations
//===----------------------------------------------------------------------===//

MLIRContextRef mlirRustContextCreate(void) {
    auto *ctx = new MLIRContext();
    ctx->loadDialect<mlir::BuiltinDialect>();
    return wrap(ctx);
}

void mlirRustContextDestroy(MLIRContextRef ctx) {
    delete unwrap(ctx);
}

void mlirRustContextEnableMultithreading(MLIRContextRef ctx, int enable) {
    unwrap(ctx)->enableMultithreading(enable != 0);
}

void mlirRustContextSetAllowUnregisteredDialects(MLIRContextRef ctx, int allow) {
    unwrap(ctx)->allowUnregisteredDialects(allow != 0);
}

void mlirRustContextLoadAllAvailableDialects(MLIRContextRef ctx) {
    unwrap(ctx)->loadAllAvailableDialects();
}

size_t mlirRustContextGetNumLoadedDialects(MLIRContextRef ctx) {
    return unwrap(ctx)->getLoadedDialects().size();
}

//===----------------------------------------------------------------------===//
// Dialect registry operations
//===----------------------------------------------------------------------===//

MLIRDialectRegistryRef mlirRustDialectRegistryCreate(void) {
    return wrap(new DialectRegistry());
}

void mlirRustDialectRegistryDestroy(MLIRDialectRegistryRef registry) {
    delete unwrap(registry);
}

void mlirRustContextAppendDialectRegistry(MLIRContextRef ctx,
                                          MLIRDialectRegistryRef registry) {
    unwrap(ctx)->appendDialectRegistry(*unwrap(registry));
}

//===----------------------------------------------------------------------===//
// Location operations
//===----------------------------------------------------------------------===//

MLIRLocationRef mlirRustLocationUnknownGet(MLIRContextRef ctx) {
    return wrap(allocLocation(UnknownLoc::get(unwrap(ctx))));
}

MLIRLocationRef mlirRustLocationFileLineColGet(MLIRContextRef ctx,
                                               MLIRStringRef filename,
                                               unsigned line,
                                               unsigned col) {
    auto fileId = StringAttr::get(unwrap(ctx), unwrapStringRef(filename));
    return wrap(allocLocation(FileLineColLoc::get(fileId, line, col)));
}

MLIRLocationRef mlirRustLocationNameGet(MLIRContextRef ctx,
                                        MLIRStringRef name,
                                        MLIRLocationRef childLoc) {
    auto nameAttr = StringAttr::get(unwrap(ctx), unwrapStringRef(name));
    Location child = childLoc ? *unwrap(childLoc) : UnknownLoc::get(unwrap(ctx));
    return wrap(allocLocation(NameLoc::get(nameAttr, child)));
}

//===----------------------------------------------------------------------===//
// Module operations
//===----------------------------------------------------------------------===//

MLIRModuleRef mlirRustModuleCreateEmpty(MLIRLocationRef loc) {
    auto module = ModuleOp::create(*unwrap(loc));
    return wrap(new ModuleOp(module));
}

MLIRModuleRef mlirRustModuleCreateParse(MLIRContextRef ctx,
                                        MLIRStringRef module_str) {
    auto result = parseSourceString<ModuleOp>(unwrapStringRef(module_str), unwrap(ctx));
    if (!result) {
        return nullptr;
    }
    return wrap(new ModuleOp(*result));
}

void mlirRustModuleDestroy(MLIRModuleRef module) {
    if (module) {
        unwrap(module)->erase();
        delete unwrap(module);
    }
}

MLIRBlockRef mlirRustModuleGetBody(MLIRModuleRef module) {
    return wrap(unwrap(module)->getBody());
}

MLIROperationRef mlirRustModuleGetOperation(MLIRModuleRef module) {
    return wrap(unwrap(module)->getOperation());
}

MLIRContextRef mlirRustModuleGetContext(MLIRModuleRef module) {
    return wrap(unwrap(module)->getContext());
}

char *mlirRustModulePrint(MLIRModuleRef module) {
    std::string str;
    llvm::raw_string_ostream os(str);
    unwrap(module)->print(os);
    char *result = static_cast<char *>(malloc(str.size() + 1));
    memcpy(result, str.c_str(), str.size() + 1);
    return result;
}

MLIRRustResult mlirRustModuleVerify(MLIRModuleRef module) {
    return mlir::verify(*unwrap(module)).succeeded()
        ? MLIRRustResult_Success
        : MLIRRustResult_Failure;
}

//===----------------------------------------------------------------------===//
// OpBuilder operations
//===----------------------------------------------------------------------===//

MLIROpBuilderRef mlirRustOpBuilderCreate(MLIRContextRef ctx) {
    return wrap(new OpBuilder(unwrap(ctx)));
}

MLIROpBuilderRef mlirRustOpBuilderCreateAtBlockEnd(MLIRBlockRef block) {
    return wrap(new OpBuilder(unwrap(block), unwrap(block)->end()));
}

void mlirRustOpBuilderDestroy(MLIROpBuilderRef builder) {
    delete unwrap(builder);
}

void mlirRustOpBuilderSetInsertionPointToEnd(MLIROpBuilderRef builder,
                                             MLIRBlockRef block) {
    unwrap(builder)->setInsertionPointToEnd(unwrap(block));
}

void mlirRustOpBuilderSetInsertionPointToStart(MLIROpBuilderRef builder,
                                               MLIRBlockRef block) {
    unwrap(builder)->setInsertionPointToStart(unwrap(block));
}

void mlirRustOpBuilderSetInsertionPoint(MLIROpBuilderRef builder,
                                        MLIROperationRef op) {
    unwrap(builder)->setInsertionPoint(unwrap(op));
}

void mlirRustOpBuilderSetInsertionPointAfter(MLIROpBuilderRef builder,
                                             MLIROperationRef op) {
    unwrap(builder)->setInsertionPointAfter(unwrap(op));
}

MLIRContextRef mlirRustOpBuilderGetContext(MLIROpBuilderRef builder) {
    return wrap(unwrap(builder)->getContext());
}

MLIRBlockRef mlirRustOpBuilderGetInsertionBlock(MLIROpBuilderRef builder) {
    return wrap(unwrap(builder)->getInsertionBlock());
}

//===----------------------------------------------------------------------===//
// Type operations
//===----------------------------------------------------------------------===//

MLIRContextRef mlirRustTypeGetContext(MLIRTypeRef type) {
    return wrap(unwrap(type)->getContext());
}

int mlirRustTypeIsInteger(MLIRTypeRef type) {
    return isa<IntegerType>(*unwrap(type));
}

int mlirRustTypeIsFloat(MLIRTypeRef type) {
    return isa<FloatType>(*unwrap(type));
}

int mlirRustTypeIsIndex(MLIRTypeRef type) {
    return isa<IndexType>(*unwrap(type));
}

MLIRTypeRef mlirRustIntegerTypeGet(MLIRContextRef ctx, unsigned width) {
    return wrap(allocType(IntegerType::get(unwrap(ctx), width)));
}

MLIRTypeRef mlirRustIntegerTypeSignlessGet(MLIRContextRef ctx, unsigned width) {
    return wrap(allocType(IntegerType::get(unwrap(ctx), width, IntegerType::Signless)));
}

MLIRTypeRef mlirRustIntegerTypeSignedGet(MLIRContextRef ctx, unsigned width) {
    return wrap(allocType(IntegerType::get(unwrap(ctx), width, IntegerType::Signed)));
}

MLIRTypeRef mlirRustIntegerTypeUnsignedGet(MLIRContextRef ctx, unsigned width) {
    return wrap(allocType(IntegerType::get(unwrap(ctx), width, IntegerType::Unsigned)));
}

MLIRTypeRef mlirRustIndexTypeGet(MLIRContextRef ctx) {
    return wrap(allocType(IndexType::get(unwrap(ctx))));
}

MLIRTypeRef mlirRustF16TypeGet(MLIRContextRef ctx) {
    return wrap(allocType(Float16Type::get(unwrap(ctx))));
}

MLIRTypeRef mlirRustBF16TypeGet(MLIRContextRef ctx) {
    return wrap(allocType(BFloat16Type::get(unwrap(ctx))));
}

MLIRTypeRef mlirRustF32TypeGet(MLIRContextRef ctx) {
    return wrap(allocType(Float32Type::get(unwrap(ctx))));
}

MLIRTypeRef mlirRustF64TypeGet(MLIRContextRef ctx) {
    return wrap(allocType(Float64Type::get(unwrap(ctx))));
}

MLIRTypeRef mlirRustNoneTypeGet(MLIRContextRef ctx) {
    return wrap(allocType(NoneType::get(unwrap(ctx))));
}

//===----------------------------------------------------------------------===//
// Attribute operations
//===----------------------------------------------------------------------===//

MLIRAttributeRef mlirRustUnitAttrGet(MLIRContextRef ctx) {
    return wrap(allocAttribute(UnitAttr::get(unwrap(ctx))));
}

MLIRAttributeRef mlirRustBoolAttrGet(MLIRContextRef ctx, int value) {
    return wrap(allocAttribute(BoolAttr::get(unwrap(ctx), value != 0)));
}

MLIRAttributeRef mlirRustIntegerAttrGet(MLIRTypeRef type, int64_t value) {
    return wrap(allocAttribute(IntegerAttr::get(*unwrap(type), value)));
}

MLIRAttributeRef mlirRustFloatAttrGet(MLIRTypeRef type, double value) {
    return wrap(allocAttribute(FloatAttr::get(*unwrap(type), value)));
}

MLIRAttributeRef mlirRustStringAttrGet(MLIRContextRef ctx, MLIRStringRef value) {
    return wrap(allocAttribute(StringAttr::get(unwrap(ctx), unwrapStringRef(value))));
}

MLIRAttributeRef mlirRustTypeAttrGet(MLIRTypeRef type) {
    return wrap(allocAttribute(TypeAttr::get(*unwrap(type))));
}

MLIRAttributeRef mlirRustFlatSymbolRefAttrGet(MLIRContextRef ctx,
                                              MLIRStringRef symbol) {
    return wrap(allocAttribute(FlatSymbolRefAttr::get(unwrap(ctx), unwrapStringRef(symbol))));
}

//===----------------------------------------------------------------------===//
// Operation state operations
//===----------------------------------------------------------------------===//

MLIROperationStateRef mlirRustOperationStateCreate(MLIRStringRef name,
                                                   MLIRLocationRef loc) {
    auto *state = new OperationState(*unwrap(loc), unwrapStringRef(name));
    return wrap(state);
}

void mlirRustOperationStateDestroy(MLIROperationStateRef state) {
    delete unwrap(state);
}

void mlirRustOperationStateAddResults(MLIROperationStateRef state,
                                      size_t n,
                                      MLIRTypeRef const *results) {
    for (size_t i = 0; i < n; ++i) {
        unwrap(state)->addTypes(*unwrap(results[i]));
    }
}

void mlirRustOperationStateAddOperands(MLIROperationStateRef state,
                                       size_t n,
                                       MLIRValueRef const *operands) {
    for (size_t i = 0; i < n; ++i) {
        unwrap(state)->addOperands(*unwrap(operands[i]));
    }
}

void mlirRustOperationStateAddAttribute(MLIROperationStateRef state,
                                        MLIRStringRef name,
                                        MLIRAttributeRef attr) {
    unwrap(state)->addAttribute(unwrapStringRef(name), *unwrap(attr));
}

void mlirRustOperationStateAddOwnedRegions(MLIROperationStateRef state,
                                           size_t n,
                                           MLIRRegionRef const *regions) {
    for (size_t i = 0; i < n; ++i) {
        unwrap(state)->addRegion(std::unique_ptr<Region>(unwrap(regions[i])));
    }
}

void mlirRustOperationStateAddSuccessors(MLIROperationStateRef state,
                                         size_t n,
                                         MLIRBlockRef const *successors) {
    for (size_t i = 0; i < n; ++i) {
        unwrap(state)->addSuccessors(unwrap(successors[i]));
    }
}

//===----------------------------------------------------------------------===//
// Operation operations
//===----------------------------------------------------------------------===//

MLIROperationRef mlirRustOperationCreate(MLIROperationStateRef state) {
    return wrap(Operation::create(*unwrap(state)));
}

void mlirRustOperationDestroy(MLIROperationRef op) {
    unwrap(op)->erase();
}

size_t mlirRustOperationGetNumResults(MLIROperationRef op) {
    return unwrap(op)->getNumResults();
}

MLIRValueRef mlirRustOperationGetResult(MLIROperationRef op, size_t pos) {
    return wrap(allocValue(unwrap(op)->getResult(pos)));
}

size_t mlirRustOperationGetNumOperands(MLIROperationRef op) {
    return unwrap(op)->getNumOperands();
}

MLIRValueRef mlirRustOperationGetOperand(MLIROperationRef op, size_t pos) {
    return wrap(allocValue(unwrap(op)->getOperand(pos)));
}

MLIRBlockRef mlirRustOperationGetBlock(MLIROperationRef op) {
    return wrap(unwrap(op)->getBlock());
}

size_t mlirRustOperationGetNumRegions(MLIROperationRef op) {
    return unwrap(op)->getNumRegions();
}

MLIRRegionRef mlirRustOperationGetRegion(MLIROperationRef op, size_t pos) {
    return wrap(&unwrap(op)->getRegion(pos));
}

char *mlirRustOperationPrint(MLIROperationRef op) {
    std::string str;
    llvm::raw_string_ostream os(str);
    unwrap(op)->print(os);
    char *result = static_cast<char *>(malloc(str.size() + 1));
    memcpy(result, str.c_str(), str.size() + 1);
    return result;
}

//===----------------------------------------------------------------------===//
// Block operations
//===----------------------------------------------------------------------===//

MLIRBlockRef mlirRustBlockCreate(size_t nArgs, MLIRTypeRef const *argTypes,
                                 MLIRLocationRef const *argLocs) {
    SmallVector<Type, 4> types;
    SmallVector<Location, 4> locs;
    for (size_t i = 0; i < nArgs; ++i) {
        types.push_back(*unwrap(argTypes[i]));
        locs.push_back(*unwrap(argLocs[i]));
    }
    return wrap(new Block());
}

void mlirRustBlockDestroy(MLIRBlockRef block) {
    delete unwrap(block);
}

size_t mlirRustBlockGetNumArguments(MLIRBlockRef block) {
    return unwrap(block)->getNumArguments();
}

MLIRValueRef mlirRustBlockGetArgument(MLIRBlockRef block, size_t pos) {
    return wrap(allocValue(unwrap(block)->getArgument(pos)));
}

MLIRValueRef mlirRustBlockAddArgument(MLIRBlockRef block, MLIRTypeRef type,
                                      MLIRLocationRef loc) {
    return wrap(allocValue(unwrap(block)->addArgument(*unwrap(type), *unwrap(loc))));
}

MLIROperationRef mlirRustBlockGetFirstOperation(MLIRBlockRef block) {
    if (unwrap(block)->empty())
        return nullptr;
    return wrap(&unwrap(block)->front());
}

MLIROperationRef mlirRustBlockGetTerminator(MLIRBlockRef block) {
    return wrap(unwrap(block)->getTerminator());
}

void mlirRustBlockAppendOperation(MLIRBlockRef block, MLIROperationRef op) {
    unwrap(block)->push_back(unwrap(op));
}

void mlirRustBlockInsertOwnedOperationBefore(MLIRBlockRef block,
                                             MLIROperationRef ref,
                                             MLIROperationRef op) {
    unwrap(block)->getOperations().insert(Block::iterator(unwrap(ref)), unwrap(op));
}

//===----------------------------------------------------------------------===//
// Region operations
//===----------------------------------------------------------------------===//

MLIRRegionRef mlirRustRegionCreate(void) {
    return wrap(new Region());
}

void mlirRustRegionDestroy(MLIRRegionRef region) {
    delete unwrap(region);
}

size_t mlirRustRegionGetNumBlocks(MLIRRegionRef region) {
    return unwrap(region)->getBlocks().size();
}

MLIRBlockRef mlirRustRegionGetFirstBlock(MLIRRegionRef region) {
    if (unwrap(region)->empty())
        return nullptr;
    return wrap(&unwrap(region)->front());
}

void mlirRustRegionAppendBlock(MLIRRegionRef region, MLIRBlockRef block) {
    unwrap(region)->push_back(unwrap(block));
}

void mlirRustRegionInsertOwnedBlockBefore(MLIRRegionRef region,
                                          MLIRBlockRef ref,
                                          MLIRBlockRef block) {
    unwrap(region)->getBlocks().insert(Region::iterator(unwrap(ref)), unwrap(block));
}

//===----------------------------------------------------------------------===//
// Value operations
//===----------------------------------------------------------------------===//

MLIRTypeRef mlirRustValueGetType(MLIRValueRef value) {
    return wrap(allocType(unwrap(value)->getType()));
}

int mlirRustValueIsBlockArgument(MLIRValueRef value) {
    return isa<BlockArgument>(*unwrap(value));
}

int mlirRustValueIsOpResult(MLIRValueRef value) {
    return isa<OpResult>(*unwrap(value));
}

char *mlirRustValuePrint(MLIRValueRef value) {
    std::string str;
    llvm::raw_string_ostream os(str);
    unwrap(value)->print(os);
    char *result = static_cast<char *>(malloc(str.size() + 1));
    memcpy(result, str.c_str(), str.size() + 1);
    return result;
}

//===----------------------------------------------------------------------===//
// Memory management
//===----------------------------------------------------------------------===//

void mlirRustStringDestroy(char *str) {
    free(str);
}
