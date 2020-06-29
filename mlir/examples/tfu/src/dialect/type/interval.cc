//
// Created by aniss on 2020/6/26.
//

#include "dialect/type/interval.h"

namespace tfu {

namespace {
Iteration DynamicComputeReuse(const DynamicIterationNode* src) {
  HExpr only_iter = src->key.is_single_point();
  HExpr inter_i_min = Let::make(src->var, src->var - 1, src->range.min);
  HExpr inter_i_max = Let::make(src->var, src->var - 1, src->range.max);
  Interval iter_i_prev(inter_i_min, inter_i_max);
  Interval intersect = Interval::make_intersection(iter_i_prev, src->range);
  HExpr is_first = src->key.min == src->var;
  HExpr min = Select::make(is_first, HExpr((int64_t)0), intersect.min);
  HExpr max = Select::make(is_first, HExpr((int64_t)0), intersect.max);
  Interval reuse(min, max);
  tvm::NodePtr<DynamicIterationNode> node = tvm::make_node<DynamicIterationNode>();
  node->var = src->var;
  node->key = src->key;
  node->range = reuse;
  return Iteration(node);
}

Iteration StaticComputeReuse(const StaticIterationNode* src) {

}

} // namespace

Iteration ComputeReuse(Iteration iter) {
  if (auto node = iter.as<DynamicIterationNode>()) {
    return DynamicComputeReuse(node);
  }
  if (auto node = iter.as<StaticIterationNode>()) {
    return StaticComputeReuse(node);
  }
}

namespace {
Iteration StaticSlidingWindowForward(const StaticIterationNode* node,
                                     HExpr k, HExpr s, HExpr size ) {

  return Iteration();
}

Iteration DynamicSlidingWindowForward(const DynamicIterationNode* node,
                                      HExpr k, HExpr s, HExpr size ) {

  return Iteration();
}

Iteration StaticPaddingForward(const StaticIterationNode* node,
                               HExpr pl, HExpr pr, HExpr size ) {

  return Iteration();
}

Iteration DynamicPaddingForward(const DynamicIterationNode* node,
                                HExpr pl, HExpr pr, HExpr size ) {

  return Iteration();
}

} // namespace

Iteration SlidingWindowForward(const Iteration& iteration,
                               HExpr k, HExpr s, HExpr size) {
  return Iteration();
}

Iteration PaddingForward(const Iteration& iteration,
                         HExpr pl, HExpr pr, HExpr size) {
  return Iteration();
}

Iteration EltwiseForward(const Iteration& iteration) {
  return Iteration();
}

}

