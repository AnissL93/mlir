//
// Created by aniss on 2020/6/20.
//

#ifndef LLVM_EXPR_H
#define LLVM_EXPR_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/Support/Casting.h"
#include <type_traits>

namespace mlir {

class MLIRContext;

namespace detail {
struct ExprStorage;
struct BinaryOpExprStorage;
struct SymbolExprStorage;
struct ConstantExprStorage;
} // namespace detail

enum class ExprKind {
  Add,
  /// RHS of mul is always a constant or a symbolic expression.
  Sub,
  Mul,
  /// RHS of mod is always a constant or a symbolic expression with a positive
  /// value.
  Mod,
  /// RHS of floordiv is always a constant or a symbolic expression.
  FloorDiv,
  /// RHS of ceildiv is always a constant or a symbolic expression.
  CeilDiv,
  Max,
  Min,

  /// This is a marker for the last affine binary op. The range of binary
  /// op's is expected to be this element and earlier.
  LAST_AFFINE_BINARY_OP = Min,

  /// Constant integer.
  Constant,
  /// Symbolic identifier.
  SymbolId,
};

/// Base type for affine expression.
/// Expr's are immutable value types with intuitive operators to
/// operate on chainable, lightweight compositions.
/// An Expr is an interface to the underlying storage type pointer.
class Expr {
public:
  using ImplType = detail::ExprStorage;

  constexpr Expr() : expr(nullptr) {}
  /* implicit */ Expr(const ImplType *expr)
      : expr(const_cast<ImplType *>(expr)) {}

  bool operator==(Expr other) const { return expr == other.expr; }
  bool operator!=(Expr other) const { return !(*this == other); }
  bool operator==(int64_t v) const;
  bool operator!=(int64_t v) const { return !(*this == v); }
  explicit operator bool() const { return expr; }

  bool operator!() const { return expr == nullptr; }

  template <typename U> bool isa() const;
  template <typename U> U dyn_cast() const;
  template <typename U> U dyn_cast_or_null() const;
  template <typename U> U cast() const;

  MLIRContext *getContext() const;

  /// Return the classification for this type.
  ExprKind getKind() const;

  void print(raw_ostream &os) const;
  void dump() const;

  /// Returns true if this expression is made out of only symbols and
  /// constants, i.e., it does not involve dimensional identifiers.
  bool isSymbolicOrConstant() const;

  /// Returns true if this is a pure affine expression, i.e., multiplication,
  /// floordiv, ceildiv, and mod is only allowed w.r.t constants.
  bool isPure() const;

  /// Return true if the affine expression involves DimExpr `position`.
  bool isFunctionOfDim(unsigned position) const;

  /// Walk all of the Expr's in this expression in postorder.
  void walk(std::function<void(Expr)> callback) const;

  Expr operator+(int64_t v) const;
  Expr operator+(Expr other) const;
  Expr operator-() const;
  Expr operator-(int64_t v) const;
  Expr operator-(Expr other) const;
  Expr operator*(int64_t v) const;
  Expr operator*(Expr other) const;
  Expr floorDiv(uint64_t v) const;
  Expr floorDiv(Expr other) const;
  Expr ceilDiv(uint64_t v) const;
  Expr ceilDiv(Expr other) const;
  Expr operator%(uint64_t v) const;
  Expr operator%(Expr other) const;

  friend ::llvm::hash_code hash_value(Expr arg);

protected:
  ImplType *expr;
};

///  binary operation expression. An affine binary operation could be an
/// add, mul, floordiv, ceildiv, or a modulo operation. (Subtraction is
/// represented through a multiply by -1 and add.) These expressions are always
/// constructed in a simplified form. For eg., the LHS and RHS operands can't
/// both be constants. There are additional canonicalizing rules depending on
/// the op type: see checks in the constructor.
class BinaryOpExpr : public Expr {
public:
  using ImplType = detail::BinaryOpExprStorage;
  /* implicit */ BinaryOpExpr(Expr::ImplType *ptr);
  Expr getLHS() const;
  Expr getRHS() const;
};

/// A symbolic identifier appearing in an affine expression.
class SymbolExpr : public Expr {
public:
  using ImplType = detail::SymbolExprStorage;
  /* implicit */ SymbolExpr(Expr::ImplType *ptr);
  unsigned getPosition() const;
  llvm::StringRef getNameHint() const;
};

/// An integer constant appearing in affine expression.
class ConstantExpr : public Expr {
public:
  using ImplType = detail::ConstantExprStorage;
  /* implicit */ ConstantExpr(Expr::ImplType *ptr = nullptr);
  int64_t getValue() const;
};

/// Make Expr hashable.
inline ::llvm::hash_code hash_value(Expr arg) {
  return ::llvm::hash_value(arg.expr);
}

inline Expr operator+(int64_t val, Expr expr) { return expr + val; }
inline Expr operator*(int64_t val, Expr expr) { return expr * val; }
inline Expr operator-(int64_t val, Expr expr) {
  return expr * (-1) + val;
}

/// These free functions allow clients of the API to not use classes in detail.
Expr getSymbolExpr(unsigned position, llvm::StringRef name, MLIRContext *context);
Expr getConstantExpr(int64_t constant, MLIRContext *context);
Expr getBinaryOpExpr(ExprKind kind, Expr lhs,
                                 Expr rhs);


raw_ostream &operator<<(raw_ostream &os, Expr expr);

template <typename U> bool Expr::isa() const {
  if (std::is_same<U, BinaryOpExpr>::value)
    return getKind() <= ExprKind::LAST_AFFINE_BINARY_OP;
  if (std::is_same<U, SymbolExpr>::value)
    return getKind() == ExprKind::SymbolId;
  if (std::is_same<U, ConstantExpr>::value)
    return getKind() == ExprKind::Constant;
}
template <typename U> U Expr::dyn_cast() const {
  if (isa<U>())
    return U(expr);
  return U(nullptr);
}
template <typename U> U Expr::dyn_cast_or_null() const {
  return (!*this || !isa<U>()) ? U(nullptr) : U(expr);
}
template <typename U> U Expr::cast() const {
  assert(isa<U>());
  return U(expr);
}

/// Simplify an affine expression by flattening and some amount of
/// simple analysis. This has complexity linear in the number of nodes in
/// 'expr'. Returns the simplified expression, which is the same as the input
///  expression if it can't be simplified.
Expr simplifyExpr(Expr expr, unsigned numDims,
                              unsigned numSymbols);

} // namespace mlir

namespace llvm {

// Expr hash just like pointers
template <> struct DenseMapInfo<mlir::Expr> {
  static mlir::Expr getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::Expr(static_cast<mlir::Expr::ImplType *>(pointer));
  }
  static mlir::Expr getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::Expr(static_cast<mlir::Expr::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::Expr val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::Expr LHS, mlir::Expr RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

#endif // LLVM_EXPR_H
