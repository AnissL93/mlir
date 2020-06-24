//
// Created by aniss on 2020/6/22.
//

#ifndef LLVM_INFER_SHAPE_H
#define LLVM_INFER_SHAPE_H

#include "mlir/Pass/Pass.h"

namespace {

class InferShape : public mlir::PassWrapper<InferShape, mlir::FunctionPass> {

};

}

#endif // LLVM_INFER_SHAPE_H
