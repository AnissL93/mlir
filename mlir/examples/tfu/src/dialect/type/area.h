//
// Created by aniss on 2020/6/28.
//

#ifndef LLVM_AREA_H
#define LLVM_AREA_H

#include "dialect/type/interval.h"

namespace tfu {

struct LayoutNode : public Node {
  static constexpr const char* _type_key = "Layout";
  std::vector<char> area;
  TVM_DECLARE_BASE_NODE_INFO(LayoutNode, tvm::Node);
};

struct Layout : public NodeRef {

};

struct AreaNode : public Node {
  static constexpr const char* _type_key = "Area";
  std::vector<Interval> area;
  TVM_DECLARE_BASE_NODE_INFO(AreaNode, tvm::Node);
};

struct DynamicAreaNode : public AreaNode {
  static constexpr const char* _type_key = "DynamicArea";
  TVM_DECLARE_BASE_NODE_INFO(DynamicAreaNode, AreaNode);
};

struct StaticAreaNode : public AreaNode {
  static constexpr const char* _type_key = "StaticArea";
  TVM_DECLARE_BASE_NODE_INFO(StaticAreaNode, AreaNode);
};

}

#endif // LLVM_AREA_H
