//
// Created by aniss on 2020/6/20.
//

#include "dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

#include <iostream>

using namespace mlir;
using namespace mlir::tfu;

TfuDialect::TfuDialect(mlir::MLIRContext *ctx) : mlir::Dialect("tfu", ctx) {
  addOperations<
#define GET_OP_LIST
#include "src/dialect/op.cpp.inc"
  >();
}

#define GET_OP_CLASSES
#include "src/dialect/op.cpp.inc"


