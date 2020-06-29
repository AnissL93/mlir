//
// Created by aniss on 2020/6/22.
//

#ifndef LLVM_INFER_SHAPE_H
#define LLVM_INFER_SHAPE_H

#include "mlir/Pass/Pass.h"
#include "dialect/dialect.h"

namespace {

template <typename ConcretePass>
class BfsVisit : public mlir::PassWrapper<ConcretePass, mlir::FunctionPass> {
public:
  using BaseType = mlir::PassWrapper<ConcretePass, mlir::FunctionPass>;

  void runOnFunction() override {
    auto f = BaseType::getFunction();

    llvm::SmallPtrSet<mlir::Operation *, 16> op_worklist;
    f.walk([&](mlir::Operation *op) {
        op_worklist.insert(op);
    });
  }

};

}

#endif // LLVM_INFER_SHAPE_H
