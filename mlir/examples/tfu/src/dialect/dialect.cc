//
// Created by aniss on 2020/6/20.
//

#include "dialect.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "type/tfu_type.h"

#include <iostream>

using namespace mlir;
using namespace mlir::tfu;

TfuDialect::TfuDialect(mlir::MLIRContext *ctx) : mlir::Dialect("tfu", ctx) {
  addOperations<
#define GET_OP_LIST
#include "src/dialect/op.cpp.inc"
  >();
  addTypes<Region>();
  addTypes<RangeType>();
}

void TfuDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {
  Region region_type = type.cast<Region>();
  printer << "region<";
  std::ostringstream os;
  for (int i = 0; i < region_type.getShape().size(); ++i) {
     os << region_type.getShape()[i];
     os << "x";
  }
  os << region_type.getMemScope().str() << "x";
  printer << os.str();
  printer << region_type.getElemType() << ">";
}

#define GET_OP_CLASSES
#include "src/dialect/op.cpp.inc"


