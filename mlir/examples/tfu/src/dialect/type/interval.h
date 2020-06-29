//
// Created by aniss on 2020/6/26.
//

#ifndef LLVM_TFU_INTERVAL_H
#define LLVM_TFU_INTERVAL_H

#include "ir/Expr.h"
#include "ir/IROperator.h"
#include "arithmetic/Interval.h"
#include "arithmetic/Substitute.h"
#include "arithmetic/Simplify.h"

namespace tfu {
using tvm::NodeRef;
using tvm::Node;
using HalideIR::Internal::Select;
using HalideIR::VarExpr;
using HalideIR::Internal::Interval;
using HalideIR::Internal::Let;
using HalideIR::Internal::substitute;

struct IterationNode : public tvm::Node {
  HalideIR::VarExpr var;
  static constexpr const char* _type_key = "Iteration";
  TVM_DECLARE_BASE_NODE_INFO(IterationNode, tvm::Node);
};

struct StaticIterationNode : public IterationNode {
  using Bound = std::pair<int64_t, int64_t>;
  using Pattern = std::pair<int64_t, int64_t>;
  std::vector<Bound> keys;
  std::vector<Pattern> ranges;
  static constexpr const char* _type_key = "StaticIteration";
  TVM_DECLARE_NODE_TYPE_INFO(StaticIterationNode, IterationNode)
};

struct DynamicIterationNode : public IterationNode {
  Interval key, range;
  static constexpr const char* _type_key = "DynamicIteration";
  TVM_DECLARE_NODE_TYPE_INFO(DynamicIterationNode, IterationNode)
};

struct Iteration : tvm::NodeRef {
  Iteration() {}
  Iteration(tvm::NodePtr<tvm::Node> p) : tvm::NodeRef(p) {}

  using ContainerType = IterationNode;

  inline const ContainerType* get() const {
    return static_cast<const ContainerType*>(node_.get());
  }
  inline const ContainerType* operator->() const {
    return static_cast<const ContainerType*>(node_.get());
  }

  Interval GetRangeOf(HExpr val) {
    if (auto node = as<DynamicIterationNode>()) {
      HExpr min = simplify(Let::make(node->var, val, node->range.min));
      HExpr max = simplify(Let::make(node->var, val, node->range.max));
      return Interval(min, max);
    }
  }

  static Iteration makeByFactor(HalideIR::VarExpr var, int64_t size, int64_t factor) {
    tvm::NodePtr<StaticIterationNode> node = tvm::make_node<StaticIterationNode>();
    node->var = var;
    factor = std::min(factor, size);
    int64_t part = size / factor;
    int64_t rem = size - part * factor;
    node->keys.push_back(std::make_pair(0, part));
    node->ranges.push_back(std::make_pair(factor, factor));
    if (rem != 0) {
      node->keys.push_back(std::make_pair(part, part+1));
      node->ranges.push_back(std::make_pair(rem, rem));
    }
    return Iteration(node);
  }

  static Iteration makeByFactor(HalideIR::VarExpr var, HExpr size, HExpr factor) {
    tvm::NodePtr<DynamicIterationNode> node = tvm::make_node<DynamicIterationNode>();
    VarExpr fac_var("factor"), rem_var("rem"), part_var("part");
    HExpr factor_expr = HalideIR::min(factor, size);
    HExpr part = size / fac_var;
    HExpr rem = size - part_var * fac_var;
    HExpr has_rem = rem > 0;
    HExpr key_ed = Select::make(has_rem, part_var + 1, part_var);
    HExpr r_min = var * fac_var;
    r_min = Let::make(rem_var, rem, r_min);
    r_min = Let::make(part_var, part, r_min);
    r_min = Let::make(fac_var, factor_expr, r_min);
    VarExpr r_min_var("range_min");
    HExpr r_ed = r_min_var + fac_var;
    r_ed = Let::make(r_min_var, r_min, r_ed);
    Interval key(0, simplify(key_ed));
    Interval range(simplify(r_min), simplify(r_ed));
    node->var = var;
    node->key = std::move(key);
    node->range = std::move(range);
    return Iteration(node);
  }
};

inline std::ostream& operator<<(std::ostream& os, const Iteration& iter) {
  if (auto node  = iter.as<StaticIterationNode>()) {
    os << "var = " << node->var << ", ";
    os << std::endl;
    size_t size = node->keys.size();
    for (size_t i = 0; i < size; ++i) {
      os << "  [" << node->keys[i].first << ", " << node->keys[i].second << ")";
      os << " --> ";
      os << "[" << node->ranges[i].first << ", " << node->ranges[i].second << ")";
      if (i != size -1) {
        os << "\n";
      }
    }
  }
  if (auto node = iter.as<DynamicIterationNode>()) {
    os << "var = " << node->var << ", ";
    os << std::endl;
    os << "  [" << node->key.min << ", " << node->key.max << ")";
    os << " --> ";
    os << "[" << node->range.min << ", " << node->range.max << ")";
  }
  return os;
}

Iteration ComputeReuse(Iteration iter);

Iteration SlidingWindowForward(const Iteration& iteration,
                               HExpr k, HExpr s, HExpr size);

Iteration PaddingForward(const Iteration& iteration,
                         HExpr pl, HExpr pr, HExpr size);

Iteration EltwiseForward(const Iteration& iteration);

} // namespace tfu

#endif // LLVM_TFU_INTERVAL_H
