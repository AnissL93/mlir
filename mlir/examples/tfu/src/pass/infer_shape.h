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

    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    f.walk([&](mlir::Operation *op) {
      if (returnsDynamicShape(op))
        opWorklist.insert(op);
    });

  }

  bool returnsDynamicShape(mlir::Operation* op) {
    if (op->getNumResults() == 0)
      return false;

    mlir::OpResult res = op->getOpResult(0);
    mlir::Type t = res.getType();
    mlir::tfu::Region region = t.dyn_cast_or_null<mlir::tfu::Region>();
    assert(region && "return type is not region");
    return true;
//    return region.isDynamic();
  }
};

}

#endif // LLVM_INFER_SHAPE_H
