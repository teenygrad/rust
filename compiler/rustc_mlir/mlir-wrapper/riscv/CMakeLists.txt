# RISC-V-specific MLIR dialects and support code (not Triton's own dialects --
# see Dialect/RVV/IR/RVVDialect.td for why RVV lives here rather than in
# src/triton/third_party).
add_subdirectory(Dialect/RVV/IR)
