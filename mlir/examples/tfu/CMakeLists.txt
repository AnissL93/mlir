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

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

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
        )

