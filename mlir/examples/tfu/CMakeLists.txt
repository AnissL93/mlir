add_custom_target(Tfu)
set_target_properties(Tfu PROPERTIES FOLDER Examples)

macro(add_tfu name)
    add_dependencies(Tfu ${name})
    add_llvm_example(${name} ${ARGN})
endmacro(add_tfu name)

include_directories(src)
add_subdirectory(src)

set(LLVM_LINK_COMPONENTS
        Core
        Support
        )

file(GLOB_RECURSE SRCS src/*.cc)

add_tfu(test-expr
        unittests/test_expr.cc
        ${SRCS}

        DEPENDS
        OpIncGen)

add_tfu(main
        src/main.cc
        ${SRCS}

        DEPENDS
        OpIncGen)

add_tfu(test-halide
        unittests/test_halide.cc)

add_tfu(test-interval
        unittests/test_interval.cc)

add_tfu(test-pass
        unittests/test_pass.cc)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# compile halide
add_subdirectory(3rdparty/HalideIR)
include_directories(3rdparty/HalideIR/src)

set(CMAKE_CXX_FLAGS "-fexceptions")

target_link_libraries(main
        PRIVATE
        ${dialect_libs}
        ${conversion_libs}
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRSideEffects
        MLIRTargetLLVMIR
        MLIRTransforms
        halideir
        )

target_link_libraries(test-expr
        PRIVATE
        gtest
        ${dialect_libs}
        ${conversion_libs}
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRSideEffects
        MLIRTargetLLVMIR
        MLIRTransforms
        halideir
        )

target_link_libraries(test-pass
        PRIVATE
        gtest
        ${dialect_libs}
        ${conversion_libs}
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRSideEffects
        MLIRTargetLLVMIR
        MLIRTransforms
        )

target_link_libraries(test-halide
        PRIVATE
        halideir )

target_link_libraries(test-interval
        PRIVATE
        halideir)

