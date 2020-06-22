set(LLVM_TARGET_DEFINITIONS op.td)
mlir_tablegen(op.h.inc -gen-op-decls)
mlir_tablegen(op.cpp.inc -gen-op-defs)
add_public_tablegen_target(OpIncGen)
