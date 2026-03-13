cmake_minimum_required(VERSION 3.20)

# Enable compile commands for IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(llvm-wrapper C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Get the project directory (parent of llvm-wrapper)
get_filename_component(PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
get_filename_component(TARGET_DIR "${PROJECT_DIR}/../../target" ABSOLUTE)

# Paths based on build.rs configuration (can be overridden via CMake variables)
if(NOT DEFINED LLVM_BUILD_DIR)
    set(LLVM_BUILD_DIR "${TARGET_DIR}/build/llvm-build/build")
endif()
if(NOT DEFINED LLVM_INSTALL_DIR)
    set(LLVM_INSTALL_DIR "${TARGET_DIR}/install")
endif()
if(NOT DEFINED LLVM_CONFIG)
    set(LLVM_CONFIG "${LLVM_INSTALL_DIR}/bin/llvm-config")
endif()
if(NOT DEFINED MLIR_DIR)
    set(MLIR_DIR "${LLVM_BUILD_DIR}/lib/cmake/mlir")
endif()
if(NOT DEFINED LLD_DIR)
    set(LLD_DIR "${LLVM_BUILD_DIR}/lib/cmake/lld")
endif()

# Triton paths (can be overridden via CMake variables)
if(NOT DEFINED TRITON_SOURCE_DIR)
    set(TRITON_SOURCE_DIR "${PROJECT_DIR}/triton")
endif()
if(NOT DEFINED TRITON_BUILD_DIR)
    set(TRITON_BUILD_DIR "${TARGET_DIR}/build/triton-build/build")
endif()
if(NOT DEFINED TRITON_LINK_DIR)
    set(TRITON_LINK_DIR "${TRITON_BUILD_DIR}")
endif()

# Check if LLVM is built
if(NOT EXISTS "${LLVM_CONFIG}")
    message(FATAL_ERROR "LLVM not found. Please build LLVM first using build.rs or set LLVM_BUILD_DIR manually.")
endif()

# Find LLVM
find_package(LLVM REQUIRED CONFIG PATHS "${LLVM_BUILD_DIR}/lib/cmake/llvm")

# Find MLIR
find_package(MLIR REQUIRED CONFIG PATHS "${MLIR_DIR}")

# Add LLVM and MLIR to module path
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# Include LLVM and MLIR utilities
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Get LLVM CXX flags using llvm-config
execute_process(
    COMMAND ${LLVM_CONFIG} --cxxflags
    OUTPUT_VARIABLE LLVM_CXXFLAGS_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Parse LLVM components
execute_process(
    COMMAND ${LLVM_CONFIG} --components
    OUTPUT_VARIABLE LLVM_COMPONENTS_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Required components
# set(REQUIRED_COMPONENTS )
# set(OPTIONAL_COMPONENTS x86 arm aarch64 amdgpu avr loongarch m68k csky mips powerpc systemz webassembly msp430 sparc nvptx hexagon riscv xtensa bpf)

# Filter components
separate_arguments(LLVM_COMPONENTS_LIST UNIX_COMMAND "${LLVM_COMPONENTS_OUTPUT}")
set(LLVM_COMPONENTS "")
foreach(COMPONENT ${LLVM_COMPONENTS_LIST})
    if(COMPONENT IN_LIST REQUIRED_COMPONENTS OR COMPONENT IN_LIST OPTIONAL_COMPONENTS)
        list(APPEND LLVM_COMPONENTS ${COMPONENT})
    endif()
endforeach()

# Verify required components
foreach(COMPONENT ${REQUIRED_COMPONENTS})
    if(NOT COMPONENT IN_LIST LLVM_COMPONENTS)
        message(FATAL_ERROR "Required LLVM component ${COMPONENT} not found")
    endif()
endforeach()

# Store component macros and optional defines for later use
set(LLVM_COMPONENT_DEFINES "")
foreach(COMPONENT ${LLVM_COMPONENTS})
    string(TOUPPER ${COMPONENT} COMPONENT_UPPER)
    list(APPEND LLVM_COMPONENT_DEFINES "LLVM_COMPONENT_${COMPONENT_UPPER}")
endforeach()

# Optional defines
set(OPTIONAL_DEFINES "")
if(DEFINED ENV{LLVM_ENZYME})
    list(APPEND OPTIONAL_DEFINES "ENZYME")
endif()

if(DEFINED ENV{LLVM_RUSTLLVM})
    list(APPEND OPTIONAL_DEFINES "LLVM_RUSTLLVM")
endif()

if(NOT DEFINED ENV{LLVM_ASSERTIONS})
    list(APPEND OPTIONAL_DEFINES "NDEBUG")
endif()

# Set compiler flags from LLVM
# Parse CXX flags and add them
separate_arguments(LLVM_CXXFLAGS_LIST UNIX_COMMAND "${LLVM_CXXFLAGS_RAW}")
set(LLVM_CXXFLAGS "")
set(LLVM_INCLUDE_DIRS_FROM_CONFIG "")
foreach(FLAG ${LLVM_CXXFLAGS_LIST})
    # Skip LTO flags
    if(FLAG MATCHES "^-flto")
        continue()
    endif()
    # Collect include directories
    if(FLAG MATCHES "^-I")
        string(REGEX REPLACE "^-I" "" INCLUDE_DIR "${FLAG}")
        list(APPEND LLVM_INCLUDE_DIRS_FROM_CONFIG "${INCLUDE_DIR}")
    else()
        # Add other flags
        list(APPEND LLVM_CXXFLAGS "${FLAG}")
    endif()
endforeach()

# Store include directories for target
set(LLVM_WRAPPER_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${LLVM_INCLUDE_DIRS}"
    ${LLVM_INCLUDE_DIRS_FROM_CONFIG}
    "${MLIR_INCLUDE_DIRS}"
    # Triton include directories
    "${TRITON_SOURCE_DIR}/include"
    "${TRITON_SOURCE_DIR}/third_party"
    "${TRITON_BUILD_DIR}/include"
    "${TRITON_BUILD_DIR}/third_party"
)

# Source files (matching build.rs)
set(LLVM_WRAPPER_SOURCES
    src/PassWrapper.cpp
    src/RustWrapper.cpp
    src/MLIRWrapper.cpp
    src/CoverageMappingWrapper.cpp
    src/SymbolWrapper.cpp
    src/Linker.cpp
)

# Create library target
add_library(llvm-wrapper STATIC ${LLVM_WRAPPER_SOURCES})
install(TARGETS llvm-wrapper
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Set target properties
set_target_properties(llvm-wrapper PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Use LLVM's CMake utilities to set up the target
llvm_update_compile_flags(llvm-wrapper)

# Set include directories
target_include_directories(llvm-wrapper PRIVATE ${LLVM_WRAPPER_INCLUDE_DIRS})

# Apply LLVM CXX flags
target_compile_options(llvm-wrapper PRIVATE ${LLVM_CXXFLAGS})

# Apply component and optional defines
target_compile_definitions(llvm-wrapper PRIVATE ${LLVM_COMPONENT_DEFINES} ${OPTIONAL_DEFINES})

# Disable warnings (as in build.rs) - but only if not in CI or MSVC
if(NOT DEFINED ENV{CI} OR MSVC)
    target_compile_options(llvm-wrapper PRIVATE -Wno-everything)
endif()

# Link LLVM libraries using llvm_map_components_to_libnames
# Get LLVM link mode
if(DEFINED ENV{LLVM_LINK_SHARED})
    set(LLVM_LINK_ARG "--link-shared")
else()
    set(LLVM_LINK_ARG "--link-static")
endif()

# Map components to library names
llvm_map_components_to_libnames(LLVM_LIBS ${LLVM_COMPONENTS})

# Filter out LLVMLineEditor as in build.rs
list(REMOVE_ITEM LLVM_LIBS LLVMLineEditor)

# Link LLVM libraries
target_link_libraries(llvm-wrapper PRIVATE ${LLVM_LIBS})

# Get LLVM ldflags for system libraries
execute_process(
    COMMAND ${LLVM_CONFIG} ${LLVM_LINK_ARG} --ldflags
    OUTPUT_VARIABLE LLVM_LDFLAGS_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Parse LLVM ldflags
separate_arguments(LLVM_LDFLAGS_LIST UNIX_COMMAND "${LLVM_LDFLAGS_OUTPUT}")
foreach(FLAG ${LLVM_LDFLAGS_LIST})
    if(FLAG MATCHES "^-L")
        string(REGEX REPLACE "^-L" "" LIB_DIR "${FLAG}")
        target_link_directories(llvm-wrapper PRIVATE "${LIB_DIR}")
    elseif(FLAG MATCHES "^-LIBPATH:")
        string(REGEX REPLACE "^-LIBPATH:" "" LIB_DIR "${FLAG}")
        target_link_directories(llvm-wrapper PRIVATE "${LIB_DIR}")
    elseif(FLAG MATCHES "^-l")
        string(REGEX REPLACE "^-l" "" LIB_NAME "${FLAG}")
        target_link_libraries(llvm-wrapper PRIVATE ${LIB_NAME})
    endif()
endforeach()

# Link Triton library
target_link_directories(llvm-wrapper PRIVATE "${TRITON_LINK_DIR}")
target_link_libraries(llvm-wrapper PRIVATE triton)

# MLIR libraries (from build.rs)
set(MLIR_LIBS
    MLIRNVVMToLLVM
    MLIRCAPIIR
    MLIRCallInterfaces
    MLIRIndexingMapOpInterface
    MLIRControlFlowInterfaces
    MLIRFunctionInterfaces
    MLIRInferTypeOpInterface
    MLIRArithUtils
    MLIRCastInterfaces
    MLIRInferIntRangeInterface
    MLIRShapedOpInterfaces
    MLIRLoopLikeInterface
    MLIRSCFDialect
    MLIRInferIntRangeCommon
    MLIRArithDialect
    MLIRMathDialect
    MLIRSCFTransforms
    MLIRControlFlowDialect
    MLIRUBDialect
    MLIRDialectUtils
    MLIRIR
    MLIRSupport
    MLIRPass
    MLIRTensorDialect
    MLIRTensorTransforms
    MLIRViewLikeInterface
    MLIRSideEffectInterfaces
    MLIRParallelCombiningOpInterface
    MLIRDestinationStyleOpInterface
    MLIRComplexDialect
    MLIRAffineDialect
    MLIRMemRefDialect
    MLIRTargetLLVMIRImport
    MLIRLLVMDialect
    MLIRDataLayoutInterfaces
    MLIRDLTIDialect
    MLIRMemorySlotInterfaces
    MLIRTransforms
    MLIRSCFToControlFlow
    MLIRControlFlowToLLVM
    MLIRIndexToLLVM
    MLIRArithToLLVM
    MLIRArithTransforms
    MLIRAnalysis
    MLIRTransformUtils
    MLIRMathToLLVM
    MLIRGPUDialect
    MLIRGPUToNVVMTransforms
    MLIRLLVMCommonConversion
    MLIRNVVMDialect
    MLIRIndexDialect
    MLIRFuncToLLVM
    MLIRLLVMIRTransforms
    MLIRFuncDialect
    MLIRTransformDialect
    MLIRRewrite
    MLIRVectorDialect
    MLIRVectorTransforms
    MLIRArithAttrToLLVMConversion
    MLIRAffineTransforms
    MLIRSubsetOpInterface
    MLIRGPUToGPURuntimeTransforms
    MLIRConvertToLLVMPass
    MLIRAffineUtils
    MLIRSCFTransformOps
    MLIRSCFUtils
    MLIRReconcileUnrealizedCasts
    MLIRUBToLLVM
    MLIRGPUTransformOps
    MLIRGPUTransforms
    MLIRPDLDialect
    MLIRPDLToPDLInterp
    MLIRRewritePDL
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRVectorInterfaces
    MLIRValueBoundsOpInterface
    MLIRMaskingOpInterface
    MLIRMaskableOpInterface
    MLIRConvertToLLVMInterface
    MLIRGPUUtils
    MLIRPDLInterpDialect
    MLIRPresburger
    MLIRBuiltinToLLVMIRTranslation
    MLIRGPUToLLVMIRTranslation
    MLIRLLVMToLLVMIRTranslation
    MLIRNVVMToLLVMIRTranslation
    MLIRTargetLLVMIRExport
)

target_link_libraries(llvm-wrapper PRIVATE ${MLIR_LIBS})

# System libraries handling (from build.rs)
if(UNIX AND NOT APPLE)
    # Check for 32-bit targets that need libatomic
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm" AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        target_link_libraries(llvm-wrapper PRIVATE atomic)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^mips" OR CMAKE_SYSTEM_PROCESSOR MATCHES "^powerpc" OR CMAKE_SYSTEM_PROCESSOR MATCHES "^sparc")
        target_link_libraries(llvm-wrapper PRIVATE atomic)
    endif()

    # RISC-V needs libatomic
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv")
        target_link_libraries(llvm-wrapper PRIVATE atomic)
    endif()

    # NetBSD specific
    if(CMAKE_SYSTEM_NAME MATCHES "NetBSD")
        target_link_libraries(llvm-wrapper PRIVATE z execinfo)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^i[56]86")
            target_link_libraries(llvm-wrapper PRIVATE atomic)
        endif()
    endif()

    # Other Unix systems
    if(CMAKE_SYSTEM_NAME MATCHES "Haiku|Darwin|Cygwin")
        target_link_libraries(llvm-wrapper PRIVATE z)
    endif()
endif()

# C++ standard library handling
if(NOT MSVC)
    if(DEFINED ENV{LLVM_STATIC_STDCPP})
        set(STDCPP_NAME "stdc++")
        if(CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "sparc64")
                set(STDCPP_NAME "estdc++")
            else()
                set(STDCPP_NAME "c++")
            endif()
        elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin|FreeBSD|AIX")
            set(STDCPP_NAME "c++")
        endif()
        target_link_libraries(llvm-wrapper PRIVATE ${STDCPP_NAME})
    elseif(LLVM_CXXFLAGS_RAW MATCHES "stdlib=libc\\+\\+")
        target_link_libraries(llvm-wrapper PRIVATE c++)
    else()
        set(STDCPP_NAME "stdc++")
        if(CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "sparc64")
                set(STDCPP_NAME "estdc++")
            else()
                set(STDCPP_NAME "c++")
            endif()
        elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin|FreeBSD|AIX")
            set(STDCPP_NAME "c++")
        endif()
        target_link_libraries(llvm-wrapper PRIVATE ${STDCPP_NAME})
    endif()
endif()

# AIX specific
if(CMAKE_SYSTEM_NAME MATCHES "AIX")
    target_link_libraries(llvm-wrapper PRIVATE c++abi unwind)
endif()

# MinGW specific
if(MINGW)
    target_link_libraries(llvm-wrapper PRIVATE shell32 uuid pthread)
endif()

# Print configuration summary
message(STATUS "LLVM-Wrapper CMake Configuration:")
message(STATUS "  LLVM_BUILD_DIR: ${LLVM_BUILD_DIR}")
message(STATUS "  LLVM_INSTALL_DIR: ${LLVM_INSTALL_DIR}")
message(STATUS "  TRITON_SOURCE_DIR: ${TRITON_SOURCE_DIR}")
message(STATUS "  TRITON_BUILD_DIR: ${TRITON_BUILD_DIR}")
message(STATUS "  MLIR_DIR: ${MLIR_DIR}")

