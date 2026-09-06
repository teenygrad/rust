set(LLVM_TARGET_DEFINITIONS RVVOps.td)
mlir_tablegen(RVVDialect.h.inc -gen-dialect-decls -dialect=rvv)
mlir_tablegen(RVVDialect.cpp.inc -gen-dialect-defs -dialect=rvv)
mlir_tablegen(RVVOps.h.inc -gen-op-decls)
mlir_tablegen(RVVOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(RVVTableGen)

add_mlir_dialect_library(MLIRRVVDialect
    Dialect.cpp

    DEPENDS
    RVVTableGen

    LINK_LIBS
    MLIRIR
    MLIRSupport
)

# LLVM/MLIR are themselves built with RTTI disabled -- they use LLVM's own
# lightweight isa<>/classof() RTTI instead of native C++ RTTI, so e.g.
# `typeinfo for mlir::Dialect` is never emitted anywhere in their libraries.
# Without this, Dialect.cpp -- which *defines* RVVDialect, a new class
# deriving from the polymorphic mlir::Dialect base -- gets compiled with
# RTTI enabled by default, which needs to reference that (nonexistent)
# base-class typeinfo and fails to link: "undefined reference to typeinfo
# for mlir::Dialect".
target_compile_options(MLIRRVVDialect PRIVATE -fno-rtti)
