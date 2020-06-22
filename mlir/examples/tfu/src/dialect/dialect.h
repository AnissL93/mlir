//
// Created by aniss on 2020/6/20.
//

#ifndef LLVM_PROJECT_MASTER_DIALECT_H
#define LLVM_PROJECT_MASTER_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/Interfaces/SideEffects.h"

namespace mlir {
namespace tfu {

class TfuDialect : public Dialect {
public:
  explicit TfuDialect(mlir::MLIRContext *ctx);

  static llvm::StringRef getDialectNamespace() { return "tfu"; }
};

#define GET_OP_CLASSES
#include "src/dialect/op.h.inc"
}
}

#endif // LLVM_PROJECT_MASTER_DIALECT_H
