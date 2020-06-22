//
// Created by aniss on 2020/6/20.
//
#ifndef TFU_SRC_DIALECT_EXPRDETAIL_H_
#define TFU_SRC_DIALECT_EXPRDETAIL_H_

#include "expr.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/StorageUniquer.h"

namespace mlir {

class MLIRContext;

namespace detail {

/// Base storage class appearing in an affine expression.
struct ExprStorage : public StorageUniquer::BaseStorage {
  MLIRContext *context;
};

struct SymbolExprStorage : public ExprStorage {
  using KeyTy = std::pair<int64_t, llvm::StringRef>;

  bool operator==(const KeyTy &key) const {
    return key.first == position && key.second == name_hint;
  }

  static SymbolExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<SymbolExprStorage>();
    result->position = key.first;
    result->name_hint = key.second;
    return result;
  }

  int64_t position;
  llvm::StringRef name_hint;
};

/// A binary operation appearing in an affine expression.
struct BinaryOpExprStorage : public ExprStorage {
  using KeyTy = std::pair<Expr, Expr>;

  bool operator==(const KeyTy &key) const {
    return key.first == lhs && key.second == rhs;
  }

  static BinaryOpExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<BinaryOpExprStorage>();
    result->lhs = key.first;
    result->rhs = key.second;
    result->context = result->lhs.getContext();
    return result;
  }

  Expr lhs;
  Expr rhs;
};

///// A dimensional or symbolic identifier appearing in an affine expression.
//struct DimExprStorage : public ExprStorage {
//  using KeyTy = unsigned;
//
//  bool operator==(const KeyTy &key) const { return position == key; }
//
//  static DimExprStorage *
//  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
//    auto *result = allocator.allocate<DimExprStorage>();
//    result->position = key;
//    return result;
//  }
//
//  /// Position of this identifier in the argument list.
//  unsigned position;
//};

/// An integer constant appearing in affine expression.
struct ConstantExprStorage : public ExprStorage {
  using KeyTy = int64_t;

  bool operator==(const KeyTy &key) const { return constant == key; }

  static ConstantExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<ConstantExprStorage>();
    result->constant = key;
    return result;
  }

  // The constant.
  int64_t constant;
};

} // end namespace detail
} // end namespace mlir
#endif // TFU_SRC_DIALECT_EXPRDETAIL_H_
