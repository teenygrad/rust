//===-- TritonWrapper.cpp - C bindings for Triton types -------------------===//
//
// Implementation of C-compatible bindings for Triton dialect types.
// This provides access to Triton-specific types and operations.
//
//===----------------------------------------------------------------------===//

#include "TritonWrapper.h"
#include "MLIRWrapper.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdlib>
#include <cstring>

// Triton headers - only included if TRITON_ENABLED is defined
#ifdef TRITON_ENABLED
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#endif

using namespace mlir;

//===----------------------------------------------------------------------===//
// Type conversion utilities (same pattern as MLIRWrapper.cpp)
//===----------------------------------------------------------------------===//

#define DEFINE_SIMPLE_CONVERSION_FUNCTIONS(TYPE, REF)                          \
    static inline TYPE *unwrap(REF ref) { return reinterpret_cast<TYPE *>(ref); }   \
    static inline REF wrap(TYPE *ptr) { return reinterpret_cast<REF>(ptr); }

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(MLIRContext, MLIRContextRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(Location, MLIRLocationRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(Operation, MLIROperationRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(Block, MLIRBlockRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(Type, MLIRTypeRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(Attribute, MLIRAttributeRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(Value, MLIRValueRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(OpBuilder, MLIROpBuilderRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(DialectRegistry, MLIRDialectRegistryRef)

static inline llvm::StringRef unwrapStringRef(MLIRStringRef str) {
    return llvm::StringRef(str.data, str.length);
}

// Storage allocators for value types
static Type *allocType(Type type) {
    return new Type(type);
}

static Attribute *allocAttribute(Attribute attr) {
    return new Attribute(attr);
}

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

MLIRRustResult tritonRustInitDialects(MLIRContextRef ctx) {
#ifdef TRITON_ENABLED
    MLIRContext *context = unwrap(ctx);
    DialectRegistry registry;

    registry.insert<mlir::triton::TritonDialect>();
    registry.insert<mlir::triton::gpu::TritonGPUDialect>();

    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();

    return MLIRRustResult_Success;
#else
    return MLIRRustResult_Failure;
#endif
}

MLIRRustResult tritonRustRegisterTritonDialect(MLIRDialectRegistryRef registry) {
#ifdef TRITON_ENABLED
    unwrap(registry)->insert<mlir::triton::TritonDialect>();
    return MLIRRustResult_Success;
#else
    return MLIRRustResult_Failure;
#endif
}

MLIRRustResult tritonRustRegisterTritonGPUDialect(MLIRDialectRegistryRef registry) {
#ifdef TRITON_ENABLED
    unwrap(registry)->insert<mlir::triton::gpu::TritonGPUDialect>();
    return MLIRRustResult_Success;
#else
    return MLIRRustResult_Failure;
#endif
}

int tritonRustIsAvailable(void) {
#ifdef TRITON_ENABLED
    return 1;
#else
    return 0;
#endif
}

//===----------------------------------------------------------------------===//
// Triton Pointer Type
//===----------------------------------------------------------------------===//

MLIRTypeRef tritonRustPointerTypeGet(MLIRTypeRef pointeeType, int addressSpace) {
#ifdef TRITON_ENABLED
    Type pointee = *unwrap(pointeeType);
    auto ptrType = mlir::triton::PointerType::get(pointee, addressSpace);
    return wrap(allocType(ptrType));
#else
    return nullptr;
#endif
}

int tritonRustTypeIsPointer(MLIRTypeRef type) {
#ifdef TRITON_ENABLED
    return isa<mlir::triton::PointerType>(*unwrap(type));
#else
    return 0;
#endif
}

MLIRTypeRef tritonRustPointerTypeGetPointeeType(MLIRTypeRef type) {
#ifdef TRITON_ENABLED
    auto ptrType = cast<mlir::triton::PointerType>(*unwrap(type));
    return wrap(allocType(ptrType.getPointeeType()));
#else
    return nullptr;
#endif
}

int tritonRustPointerTypeGetAddressSpace(MLIRTypeRef type) {
#ifdef TRITON_ENABLED
    auto ptrType = cast<mlir::triton::PointerType>(*unwrap(type));
    return ptrType.getAddressSpace();
#else
    return 0;
#endif
}

//===----------------------------------------------------------------------===//
// Triton Tensor Type
//===----------------------------------------------------------------------===//

MLIRTypeRef tritonRustRankedTensorTypeGet(size_t rank, const int64_t *shape,
                                          MLIRTypeRef elementType,
                                          MLIRAttributeRef encoding) {
    llvm::SmallVector<int64_t, 4> shapeVec(shape, shape + rank);
    Type elemType = *unwrap(elementType);

    Attribute enc = encoding ? *unwrap(encoding) : Attribute();
    auto tensorType = RankedTensorType::get(shapeVec, elemType, enc);
    return wrap(allocType(tensorType));
}

size_t tritonRustRankedTensorTypeGetRank(MLIRTypeRef type) {
    auto tensorType = cast<RankedTensorType>(*unwrap(type));
    return tensorType.getRank();
}

int64_t tritonRustRankedTensorTypeGetDimSize(MLIRTypeRef type, size_t dim) {
    auto tensorType = cast<RankedTensorType>(*unwrap(type));
    return tensorType.getDimSize(dim);
}

MLIRTypeRef tritonRustRankedTensorTypeGetElementType(MLIRTypeRef type) {
    auto tensorType = cast<RankedTensorType>(*unwrap(type));
    return wrap(allocType(tensorType.getElementType()));
}

MLIRAttributeRef tritonRustRankedTensorTypeGetEncoding(MLIRTypeRef type) {
    auto tensorType = cast<RankedTensorType>(*unwrap(type));
    Attribute enc = tensorType.getEncoding();
    if (!enc)
        return nullptr;
    return wrap(allocAttribute(enc));
}

//===----------------------------------------------------------------------===//
// Triton GPU Blocked Encoding Attribute
//===----------------------------------------------------------------------===//

MLIRAttributeRef tritonRustBlockedEncodingAttrGet(
    MLIRContextRef ctx,
    size_t rank,
    const unsigned *sizePerThread,
    const unsigned *threadsPerWarp,
    const unsigned *warpsPerCTA,
    const unsigned *order,
    const unsigned *CTAsPerCGA,
    const unsigned *CTASplitNum,
    const unsigned *CTAOrder) {
#ifdef TRITON_ENABLED
    MLIRContext *context = unwrap(ctx);

    llvm::SmallVector<unsigned, 4> sptVec(sizePerThread, sizePerThread + rank);
    llvm::SmallVector<unsigned, 4> tpwVec(threadsPerWarp, threadsPerWarp + rank);
    llvm::SmallVector<unsigned, 4> wpcVec(warpsPerCTA, warpsPerCTA + rank);
    llvm::SmallVector<unsigned, 4> orderVec(order, order + rank);

    // CTA-level attributes (optional, use defaults if not provided)
    llvm::SmallVector<unsigned, 4> ctasPerCGA;
    llvm::SmallVector<unsigned, 4> ctaSplitNum;
    llvm::SmallVector<unsigned, 4> ctaOrder;

    if (CTAsPerCGA) {
        ctasPerCGA.assign(CTAsPerCGA, CTAsPerCGA + rank);
    } else {
        ctasPerCGA.assign(rank, 1);
    }

    if (CTASplitNum) {
        ctaSplitNum.assign(CTASplitNum, CTASplitNum + rank);
    } else {
        ctaSplitNum.assign(rank, 1);
    }

    if (CTAOrder) {
        ctaOrder.assign(CTAOrder, CTAOrder + rank);
    } else {
        for (unsigned i = 0; i < rank; ++i) {
            ctaOrder.push_back(rank - 1 - i);
        }
    }

    auto ctaLayout = mlir::triton::gpu::CTALayoutAttr::get(
        context, ctasPerCGA, ctaSplitNum, ctaOrder);

    auto blockedAttr = mlir::triton::gpu::BlockedEncodingAttr::get(
        context, sptVec, tpwVec, wpcVec, orderVec, ctaLayout);

    return wrap(allocAttribute(blockedAttr));
#else
    return nullptr;
#endif
}

int tritonRustAttrIsBlockedEncoding(MLIRAttributeRef attr) {
#ifdef TRITON_ENABLED
    return isa<mlir::triton::gpu::BlockedEncodingAttr>(*unwrap(attr));
#else
    return 0;
#endif
}

//===----------------------------------------------------------------------===//
// Triton GPU Shared Encoding Attribute
//===----------------------------------------------------------------------===//

MLIRAttributeRef tritonRustSharedEncodingAttrGet(
    MLIRContextRef ctx,
    unsigned vec,
    unsigned perPhase,
    unsigned maxPhase,
    size_t orderLen,
    const unsigned *order,
    int hasLeadingOffset) {
#ifdef TRITON_ENABLED
    MLIRContext *context = unwrap(ctx);

    llvm::SmallVector<unsigned, 4> orderVec(order, order + orderLen);

    // Default CTA layout
    llvm::SmallVector<unsigned, 2> ctasPerCGA = {1, 1};
    llvm::SmallVector<unsigned, 2> ctaSplitNum = {1, 1};
    llvm::SmallVector<unsigned, 2> ctaOrder = {1, 0};

    auto ctaLayout = mlir::triton::gpu::CTALayoutAttr::get(
        context, ctasPerCGA, ctaSplitNum, ctaOrder);

    auto sharedAttr = mlir::triton::gpu::SharedEncodingAttr::get(
        context, vec, perPhase, maxPhase, orderVec, ctaLayout,
        hasLeadingOffset != 0);

    return wrap(allocAttribute(sharedAttr));
#else
    return nullptr;
#endif
}

int tritonRustAttrIsSharedEncoding(MLIRAttributeRef attr) {
#ifdef TRITON_ENABLED
    return isa<mlir::triton::gpu::SharedEncodingAttr>(*unwrap(attr));
#else
    return 0;
#endif
}

//===----------------------------------------------------------------------===//
// Triton GPU Slice Encoding Attribute
//===----------------------------------------------------------------------===//

MLIRAttributeRef tritonRustSliceEncodingAttrGet(MLIRContextRef ctx,
                                                unsigned dim,
                                                MLIRAttributeRef parent) {
#ifdef TRITON_ENABLED
    MLIRContext *context = unwrap(ctx);
    Attribute parentAttr = *unwrap(parent);

    auto sliceAttr = mlir::triton::gpu::SliceEncodingAttr::get(
        context, dim, parentAttr);

    return wrap(allocAttribute(sliceAttr));
#else
    return nullptr;
#endif
}

int tritonRustAttrIsSliceEncoding(MLIRAttributeRef attr) {
#ifdef TRITON_ENABLED
    return isa<mlir::triton::gpu::SliceEncodingAttr>(*unwrap(attr));
#else
    return 0;
#endif
}

//===----------------------------------------------------------------------===//
// Triton GPU MMA Encoding
//===----------------------------------------------------------------------===//

MLIRAttributeRef tritonRustNvidiaMmaEncodingAttrGet(
    MLIRContextRef ctx,
    unsigned versionMajor,
    unsigned versionMinor,
    size_t warpsPerCTALen,
    const unsigned *warpsPerCTA,
    size_t CTAsPerCGALen,
    const unsigned *CTAsPerCGA,
    size_t CTASplitNumLen,
    const unsigned *CTASplitNum,
    size_t CTAOrderLen,
    const unsigned *CTAOrder,
    size_t instrShapeLen,
    const unsigned *instrShape) {
#ifdef TRITON_ENABLED
    MLIRContext *context = unwrap(ctx);

    llvm::SmallVector<unsigned, 2> wpcVec(warpsPerCTA, warpsPerCTA + warpsPerCTALen);
    llvm::SmallVector<unsigned, 2> ctasPerCGA(CTAsPerCGA, CTAsPerCGA + CTAsPerCGALen);
    llvm::SmallVector<unsigned, 2> ctaSplitNum(CTASplitNum, CTASplitNum + CTASplitNumLen);
    llvm::SmallVector<unsigned, 2> ctaOrder(CTAOrder, CTAOrder + CTAOrderLen);
    llvm::SmallVector<unsigned, 2> instrShapeVec(instrShape, instrShape + instrShapeLen);

    auto ctaLayout = mlir::triton::gpu::CTALayoutAttr::get(
        context, ctasPerCGA, ctaSplitNum, ctaOrder);

    auto mmaAttr = mlir::triton::gpu::NvidiaMmaEncodingAttr::get(
        context, versionMajor, versionMinor, wpcVec, ctaLayout, instrShapeVec);

    return wrap(allocAttribute(mmaAttr));
#else
    return nullptr;
#endif
}

MLIRAttributeRef tritonRustAMDMfmaEncodingAttrGet(
    MLIRContextRef ctx,
    unsigned versionMajor,
    unsigned versionMinor,
    size_t warpsPerCTALen,
    const unsigned *warpsPerCTA,
    unsigned mDim,
    unsigned nDim,
    int isTransposed,
    size_t CTAsPerCGALen,
    const unsigned *CTAsPerCGA,
    size_t CTASplitNumLen,
    const unsigned *CTASplitNum,
    size_t CTAOrderLen,
    const unsigned *CTAOrder) {
#ifdef TRITON_ENABLED
    MLIRContext *context = unwrap(ctx);

    llvm::SmallVector<unsigned, 2> wpcVec(warpsPerCTA, warpsPerCTA + warpsPerCTALen);
    llvm::SmallVector<unsigned, 2> ctasPerCGA(CTAsPerCGA, CTAsPerCGA + CTAsPerCGALen);
    llvm::SmallVector<unsigned, 2> ctaSplitNum(CTASplitNum, CTASplitNum + CTASplitNumLen);
    llvm::SmallVector<unsigned, 2> ctaOrder(CTAOrder, CTAOrder + CTAOrderLen);

    auto ctaLayout = mlir::triton::gpu::CTALayoutAttr::get(
        context, ctasPerCGA, ctaSplitNum, ctaOrder);

    auto mfmaAttr = mlir::triton::gpu::AMDMfmaEncodingAttr::get(
        context, versionMajor, versionMinor, wpcVec, mDim, nDim,
        isTransposed != 0, ctaLayout);

    return wrap(allocAttribute(mfmaAttr));
#else
    return nullptr;
#endif
}

//===----------------------------------------------------------------------===//
// Triton Attributes
//===----------------------------------------------------------------------===//

MLIRAttributeRef tritonRustProgramIdAttrGet(MLIRContextRef ctx, int axis) {
    // Program ID is typically represented as an integer attribute
    MLIRContext *context = unwrap(ctx);
    auto intType = IntegerType::get(context, 32);
    return wrap(allocAttribute(IntegerAttr::get(intType, axis)));
}

MLIRAttributeRef tritonRustCacheModifierAttrGet(MLIRContextRef ctx, int modifier) {
#ifdef TRITON_ENABLED
    MLIRContext *context = unwrap(ctx);
    auto cacheAttr = mlir::triton::CacheModifierAttr::get(
        context, static_cast<mlir::triton::CacheModifier>(modifier));
    return wrap(allocAttribute(cacheAttr));
#else
    return nullptr;
#endif
}

MLIRAttributeRef tritonRustEvictionPolicyAttrGet(MLIRContextRef ctx, int policy) {
#ifdef TRITON_ENABLED
    MLIRContext *context = unwrap(ctx);
    auto evictAttr = mlir::triton::EvictionPolicyAttr::get(
        context, static_cast<mlir::triton::EvictionPolicy>(policy));
    return wrap(allocAttribute(evictAttr));
#else
    return nullptr;
#endif
}

MLIRAttributeRef tritonRustPaddingOptionAttrGet(MLIRContextRef ctx, int option) {
#ifdef TRITON_ENABLED
    MLIRContext *context = unwrap(ctx);
    auto padAttr = mlir::triton::PaddingOptionAttr::get(
        context, static_cast<mlir::triton::PaddingOption>(option));
    return wrap(allocAttribute(padAttr));
#else
    return nullptr;
#endif
}

MLIRAttributeRef tritonRustPropagateNanAttrGet(MLIRContextRef ctx, int option) {
#ifdef TRITON_ENABLED
    MLIRContext *context = unwrap(ctx);
    auto nanAttr = mlir::triton::PropagateNanAttr::get(
        context, static_cast<mlir::triton::PropagateNan>(option));
    return wrap(allocAttribute(nanAttr));
#else
    return nullptr;
#endif
}

MLIRAttributeRef tritonRustRMWOpAttrGet(MLIRContextRef ctx, int op) {
#ifdef TRITON_ENABLED
    MLIRContext *context = unwrap(ctx);
    auto rmwAttr = mlir::triton::RMWOpAttr::get(
        context, static_cast<mlir::triton::RMWOp>(op));
    return wrap(allocAttribute(rmwAttr));
#else
    return nullptr;
#endif
}

//===----------------------------------------------------------------------===//
// Triton operation helpers
//===----------------------------------------------------------------------===//

// Helper to create an operation and insert it at builder's current position
static Operation *createAndInsertOp(OpBuilder *builder, OperationState &state) {
    Operation *op = builder->create(state);
    return op;
}

MLIROperationRef tritonRustMakeRangeOp(MLIROpBuilderRef builder,
                                       MLIRLocationRef loc,
                                       int32_t start,
                                       int32_t end) {
#ifdef TRITON_ENABLED
    OpBuilder *b = unwrap(builder);
    Location location = *unwrap(loc);

    int32_t size = end - start;
    auto resultType = RankedTensorType::get({size}, b->getI32Type());

    OperationState state(location, "tt.make_range");
    state.addTypes(resultType);
    state.addAttribute("start", b->getI32IntegerAttr(start));
    state.addAttribute("end", b->getI32IntegerAttr(end));

    return wrap(createAndInsertOp(b, state));
#else
    return nullptr;
#endif
}

MLIROperationRef tritonRustSplatOp(MLIROpBuilderRef builder,
                                   MLIRLocationRef loc,
                                   MLIRValueRef src,
                                   MLIRTypeRef resultType) {
#ifdef TRITON_ENABLED
    OpBuilder *b = unwrap(builder);
    Location location = *unwrap(loc);

    OperationState state(location, "tt.splat");
    state.addTypes(*unwrap(resultType));
    state.addOperands(*unwrap(src));

    return wrap(createAndInsertOp(b, state));
#else
    return nullptr;
#endif
}

MLIROperationRef tritonRustBroadcastOp(MLIROpBuilderRef builder,
                                       MLIRLocationRef loc,
                                       MLIRValueRef src,
                                       MLIRTypeRef resultType) {
#ifdef TRITON_ENABLED
    OpBuilder *b = unwrap(builder);
    Location location = *unwrap(loc);

    OperationState state(location, "tt.broadcast");
    state.addTypes(*unwrap(resultType));
    state.addOperands(*unwrap(src));

    return wrap(createAndInsertOp(b, state));
#else
    return nullptr;
#endif
}

MLIROperationRef tritonRustExpandDimsOp(MLIROpBuilderRef builder,
                                        MLIRLocationRef loc,
                                        MLIRValueRef src,
                                        int axis) {
#ifdef TRITON_ENABLED
    OpBuilder *b = unwrap(builder);
    Location location = *unwrap(loc);

    // Get input type and compute output type
    auto srcType = cast<RankedTensorType>(unwrap(src)->getType());
    auto shape = srcType.getShape();
    llvm::SmallVector<int64_t, 4> newShape(shape.begin(), shape.end());
    newShape.insert(newShape.begin() + axis, 1);
    auto resultType = RankedTensorType::get(newShape, srcType.getElementType(),
                                            srcType.getEncoding());

    OperationState state(location, "tt.expand_dims");
    state.addTypes(resultType);
    state.addOperands(*unwrap(src));
    state.addAttribute("axis", b->getI32IntegerAttr(axis));

    return wrap(createAndInsertOp(b, state));
#else
    return nullptr;
#endif
}

MLIROperationRef tritonRustAddPtrOp(MLIROpBuilderRef builder,
                                    MLIRLocationRef loc,
                                    MLIRValueRef ptr,
                                    MLIRValueRef offset) {
#ifdef TRITON_ENABLED
    OpBuilder *b = unwrap(builder);
    Location location = *unwrap(loc);

    Type resultType = unwrap(ptr)->getType();

    OperationState state(location, "tt.addptr");
    state.addTypes(resultType);
    state.addOperands({*unwrap(ptr), *unwrap(offset)});

    return wrap(createAndInsertOp(b, state));
#else
    return nullptr;
#endif
}

MLIROperationRef tritonRustLoadOp(MLIROpBuilderRef builder,
                                  MLIRLocationRef loc,
                                  MLIRValueRef ptr,
                                  MLIRAttributeRef cache,
                                  MLIRAttributeRef evict,
                                  int isVolatile) {
#ifdef TRITON_ENABLED
    OpBuilder *b = unwrap(builder);
    Location location = *unwrap(loc);

    // Get the element type from the pointer type
    Type ptrType = unwrap(ptr)->getType();
    Type elemType;
    if (auto tensorPtrType = dyn_cast<RankedTensorType>(ptrType)) {
        auto ptrElemType = cast<mlir::triton::PointerType>(
            tensorPtrType.getElementType());
        elemType = RankedTensorType::get(tensorPtrType.getShape(),
                                         ptrElemType.getPointeeType(),
                                         tensorPtrType.getEncoding());
    } else if (auto ptrElemType = dyn_cast<mlir::triton::PointerType>(ptrType)) {
        elemType = ptrElemType.getPointeeType();
    }

    OperationState state(location, "tt.load");
    state.addTypes(elemType);
    state.addOperands(*unwrap(ptr));

    if (cache)
        state.addAttribute("cache", *unwrap(cache));
    if (evict)
        state.addAttribute("evict", *unwrap(evict));
    if (isVolatile)
        state.addAttribute("isVolatile", b->getUnitAttr());

    return wrap(createAndInsertOp(b, state));
#else
    return nullptr;
#endif
}

MLIROperationRef tritonRustStoreOp(MLIROpBuilderRef builder,
                                   MLIRLocationRef loc,
                                   MLIRValueRef ptr,
                                   MLIRValueRef value,
                                   MLIRAttributeRef cache) {
#ifdef TRITON_ENABLED
    OpBuilder *b = unwrap(builder);
    Location location = *unwrap(loc);

    OperationState state(location, "tt.store");
    state.addOperands({*unwrap(ptr), *unwrap(value)});

    if (cache)
        state.addAttribute("cache", *unwrap(cache));

    return wrap(createAndInsertOp(b, state));
#else
    return nullptr;
#endif
}

MLIROperationRef tritonRustDotOp(MLIROpBuilderRef builder,
                                 MLIRLocationRef loc,
                                 MLIRValueRef a,
                                 MLIRValueRef b_operand,
                                 MLIRValueRef c,
                                 int allowTF32,
                                 int maxNumImpreciseAcc) {
#ifdef TRITON_ENABLED
    OpBuilder *b = unwrap(builder);
    Location location = *unwrap(loc);

    // Result type matches C's type
    Type resultType = unwrap(c)->getType();

    OperationState state(location, "tt.dot");
    state.addTypes(resultType);
    state.addOperands({*unwrap(a), *unwrap(b_operand), *unwrap(c)});
    state.addAttribute("allowTF32", b->getBoolAttr(allowTF32 != 0));
    state.addAttribute("maxNumImpreciseAcc", b->getI32IntegerAttr(maxNumImpreciseAcc));

    return wrap(createAndInsertOp(b, state));
#else
    return nullptr;
#endif
}

MLIROperationRef tritonRustReduceOp(MLIROpBuilderRef builder,
                                    MLIRLocationRef loc,
                                    MLIRValueRef operand,
                                    int axis) {
#ifdef TRITON_ENABLED
    OpBuilder *b = unwrap(builder);
    Location location = *unwrap(loc);

    // Compute result type by removing the reduced dimension
    auto inputType = cast<RankedTensorType>(unwrap(operand)->getType());
    auto shape = inputType.getShape();
    llvm::SmallVector<int64_t, 4> newShape;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (static_cast<int>(i) != axis)
            newShape.push_back(shape[i]);
    }

    Type resultType;
    if (newShape.empty()) {
        resultType = inputType.getElementType();
    } else {
        resultType = RankedTensorType::get(newShape, inputType.getElementType(),
                                           inputType.getEncoding());
    }

    OperationState state(location, "tt.reduce");
    state.addTypes(resultType);
    state.addOperands(*unwrap(operand));
    state.addAttribute("axis", b->getI32IntegerAttr(axis));

    // Add empty region (caller must populate it with combine operations)
    state.addRegion(std::make_unique<Region>());

    return wrap(createAndInsertOp(b, state));
#else
    return nullptr;
#endif
}

MLIROperationRef tritonRustGetProgramIdOp(MLIROpBuilderRef builder,
                                          MLIRLocationRef loc,
                                          int axis) {
#ifdef TRITON_ENABLED
    OpBuilder *b = unwrap(builder);
    Location location = *unwrap(loc);

    OperationState state(location, "tt.get_program_id");
    state.addTypes(b->getI32Type());
    state.addAttribute("axis", b->getI32IntegerAttr(axis));

    return wrap(createAndInsertOp(b, state));
#else
    return nullptr;
#endif
}

MLIROperationRef tritonRustGetNumProgramsOp(MLIROpBuilderRef builder,
                                            MLIRLocationRef loc,
                                            int axis) {
#ifdef TRITON_ENABLED
    OpBuilder *b = unwrap(builder);
    Location location = *unwrap(loc);

    OperationState state(location, "tt.get_num_programs");
    state.addTypes(b->getI32Type());
    state.addAttribute("axis", b->getI32IntegerAttr(axis));

    return wrap(createAndInsertOp(b, state));
#else
    return nullptr;
#endif
}

MLIROperationRef tritonRustFuncOp(MLIROpBuilderRef builder,
                                  MLIRLocationRef loc,
                                  MLIRStringRef name,
                                  size_t numInputs,
                                  MLIRTypeRef const *inputTypes,
                                  size_t numResults,
                                  MLIRTypeRef const *resultTypes) {
#ifdef TRITON_ENABLED
    OpBuilder *b = unwrap(builder);
    Location location = *unwrap(loc);

    llvm::SmallVector<Type, 4> inputs;
    for (size_t i = 0; i < numInputs; ++i) {
        inputs.push_back(*unwrap(inputTypes[i]));
    }

    llvm::SmallVector<Type, 4> results;
    for (size_t i = 0; i < numResults; ++i) {
        results.push_back(*unwrap(resultTypes[i]));
    }

    auto funcType = b->getFunctionType(inputs, results);

    OperationState state(location, "tt.func");
    state.addAttribute("sym_name", b->getStringAttr(unwrapStringRef(name)));
    state.addAttribute("function_type", TypeAttr::get(funcType));

    // Add the entry block region
    auto *region = new Region();
    auto *block = new Block();
    for (size_t i = 0; i < numInputs; ++i) {
        block->addArgument(inputs[i], location);
    }
    region->push_back(block);
    state.addRegion(std::unique_ptr<Region>(region));

    return wrap(createAndInsertOp(b, state));
#else
    return nullptr;
#endif
}

MLIROperationRef tritonRustReturnOp(MLIROpBuilderRef builder,
                                    MLIRLocationRef loc,
                                    size_t numValues,
                                    MLIRValueRef const *values) {
#ifdef TRITON_ENABLED
    OpBuilder *b = unwrap(builder);
    Location location = *unwrap(loc);

    OperationState state(location, "tt.return");
    for (size_t i = 0; i < numValues; ++i) {
        state.addOperands(*unwrap(values[i]));
    }

    return wrap(createAndInsertOp(b, state));
#else
    return nullptr;
#endif
}

//===----------------------------------------------------------------------===//
// Triton GPU operation helpers
//===----------------------------------------------------------------------===//

MLIROperationRef tritonRustConvertLayoutOp(MLIROpBuilderRef builder,
                                           MLIRLocationRef loc,
                                           MLIRValueRef src,
                                           MLIRTypeRef dstType) {
#ifdef TRITON_ENABLED
    OpBuilder *b = unwrap(builder);
    Location location = *unwrap(loc);

    OperationState state(location, "triton_gpu.convert_layout");
    state.addTypes(*unwrap(dstType));
    state.addOperands(*unwrap(src));

    return wrap(createAndInsertOp(b, state));
#else
    return nullptr;
#endif
}

MLIROperationRef tritonRustAllocTensorOp(MLIROpBuilderRef builder,
                                         MLIRLocationRef loc,
                                         MLIRTypeRef tensorType) {
#ifdef TRITON_ENABLED
    OpBuilder *b = unwrap(builder);
    Location location = *unwrap(loc);

    OperationState state(location, "triton_gpu.alloc_tensor");
    state.addTypes(*unwrap(tensorType));

    return wrap(createAndInsertOp(b, state));
#else
    return nullptr;
#endif
}

MLIROperationRef tritonRustInsertSliceAsyncOp(MLIROpBuilderRef builder,
                                              MLIRLocationRef loc,
                                              MLIRValueRef src,
                                              MLIRValueRef dst,
                                              MLIRValueRef index,
                                              MLIRAttributeRef cache,
                                              MLIRAttributeRef evict,
                                              int isVolatile) {
#ifdef TRITON_ENABLED
    OpBuilder *b = unwrap(builder);
    Location location = *unwrap(loc);

    Type resultType = unwrap(dst)->getType();

    OperationState state(location, "triton_gpu.insert_slice_async");
    state.addTypes(resultType);
    state.addOperands({*unwrap(src), *unwrap(dst), *unwrap(index)});

    if (cache)
        state.addAttribute("cache", *unwrap(cache));
    if (evict)
        state.addAttribute("evict", *unwrap(evict));
    if (isVolatile)
        state.addAttribute("isVolatile", b->getUnitAttr());

    return wrap(createAndInsertOp(b, state));
#else
    return nullptr;
#endif
}

MLIROperationRef tritonRustAsyncWaitOp(MLIROpBuilderRef builder,
                                       MLIRLocationRef loc,
                                       int num) {
#ifdef TRITON_ENABLED
    OpBuilder *b = unwrap(builder);
    Location location = *unwrap(loc);

    OperationState state(location, "triton_gpu.async_wait");
    state.addAttribute("num", b->getI32IntegerAttr(num));

    return wrap(createAndInsertOp(b, state));
#else
    return nullptr;
#endif
}

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

const char *tritonRustGetVersion(void) {
#ifdef TRITON_ENABLED
    // Return a static version string
    return "3.6.0";
#else
    return "not available";
#endif
}

unsigned tritonRustGetNumWarps(MLIRStringRef target) {
    llvm::StringRef targetStr = unwrapStringRef(target);

    // Default values based on common GPU configurations
    if (targetStr.contains("sm_") || targetStr.contains("cuda")) {
        return 4; // NVIDIA default
    } else if (targetStr.contains("gfx") || targetStr.contains("amd")) {
        return 4; // AMD default
    }

    return 4; // Generic default
}

unsigned tritonRustGetThreadsPerWarp(MLIRStringRef target) {
    llvm::StringRef targetStr = unwrapStringRef(target);

    if (targetStr.contains("gfx") || targetStr.contains("amd")) {
        return 64; // AMD wavefront size
    }

    return 32; // NVIDIA warp size (default)
}
