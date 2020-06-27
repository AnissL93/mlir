//
// Created by aniss on 2020/6/26.
//

#ifndef LLVM_TFU_INTERVAL_H
#define LLVM_TFU_INTERVAL_H

#include "ir/Expr.h"
#include "range.h"

namespace tfu {
struct IntervalNode;
struct DynamicIntervalNode;
struct StaticIntervalNode;

class Interval : public tvm::NodeRef {
public:
  Interval() {}
  explicit Interval(tvm::NodePtr<tvm::Node> n) : tvm::NodeRef(n) {}

  inline const tvm::Node *get() const {
    return static_cast<const tvm::Node *>(node_.get());
  }

  inline const tvm::Node *operator->() const {
    return static_cast<const tvm::Node *>(node_.get());
  }
};

struct IntervalNode : public tvm::Node {
  using ContainerType =
      std::vector<std::pair<Range, tvm::NodePtr<tvm::Node>>>;

  ContainerType data;

  static constexpr const char* _type_key = "Interval";
  TVM_DECLARE_NODE_TYPE_INFO(IntervalNode, Node);
};

struct DynamicIntervalNode : public IntervalNode {

};

struct StaticIntervalNode : public IntervalNode {

};

} // namespace tfu

#endif // LLVM_TFU_INTERVAL_H
