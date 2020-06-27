//
// Created by aniss on 2020/6/26.
//

#ifndef LLVM_TFU_RANGE_H
#define LLVM_TFU_RANGE_H

#include "ir/Expr.h"

namespace tfu {

class RangeNode;

class Range : public tvm::NodeRef {
public:
  static Range make(HExpr st, HExpr ed);
};

struct RangeNode : public tvm::Node {
  HExpr start, end;

  static constexpr const char* _type_key = "TfuRange";
  TVM_DECLARE_NODE_TYPE_INFO(RangeNode, Node);
};

}

#endif // LLVM_RANGE_H
