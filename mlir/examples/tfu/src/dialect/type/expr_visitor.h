//
// Created by aniss on 2020/6/20.
//

#ifndef LLVM_EXPR_VISITOR_H
#define LLVM_EXPR_VISITOR_H

#include "expr.h"

namespace mlir {

template <typename SubClass, typename RetTy = void>
class ExprVisitor {
public:
  // Function to walk an Expr (in post order).
  RetTy walkPostOrder(Expr expr) {
    static_assert(std::is_base_of<ExprVisitor, SubClass>::value,
                  "Must instantiate with a derived type of ExprVisitor");
    switch (expr.getKind()) {
    case ExprKind::Add: {
      auto binOpExpr = expr.cast<BinaryOpExpr>();
      walkOperandsPostOrder(binOpExpr);
      return static_cast<SubClass *>(this)->visitAddExpr(binOpExpr);
    }
    case ExprKind::Mul: {
      auto binOpExpr = expr.cast<BinaryOpExpr>();
      walkOperandsPostOrder(binOpExpr);
      return static_cast<SubClass *>(this)->visitMulExpr(binOpExpr);
    }
    case ExprKind::Mod: {
      auto binOpExpr = expr.cast<BinaryOpExpr>();
      walkOperandsPostOrder(binOpExpr);
      return static_cast<SubClass *>(this)->visitModExpr(binOpExpr);
    }
    case ExprKind::FloorDiv: {
      auto binOpExpr = expr.cast<BinaryOpExpr>();
      walkOperandsPostOrder(binOpExpr);
      return static_cast<SubClass *>(this)->visitFloorDivExpr(binOpExpr);
    }
    case ExprKind::CeilDiv: {
      auto binOpExpr = expr.cast<BinaryOpExpr>();
      walkOperandsPostOrder(binOpExpr);
      return static_cast<SubClass *>(this)->visitCeilDivExpr(binOpExpr);
    }
    case ExprKind::Constant:
      return static_cast<SubClass *>(this)->visitConstantExpr(
          expr.cast<ConstantExpr>());
    case ExprKind::SymbolId:
      return static_cast<SubClass *>(this)->visitSymbolExpr(
          expr.cast<SymbolExpr>());
    }
  }

  // Function to visit an Expr.
  RetTy visit(Expr expr) {
    static_assert(std::is_base_of<ExprVisitor, SubClass>::value,
                  "Must instantiate with a derived type of ExprVisitor");
    switch (expr.getKind()) {
    case ExprKind::Add: {
      auto binOpExpr = expr.cast<BinaryOpExpr>();
      return static_cast<SubClass *>(this)->visitAddExpr(binOpExpr);
    }
    case ExprKind::Mul: {
      auto binOpExpr = expr.cast<BinaryOpExpr>();
      return static_cast<SubClass *>(this)->visitMulExpr(binOpExpr);
    }
    case ExprKind::Mod: {
      auto binOpExpr = expr.cast<BinaryOpExpr>();
      return static_cast<SubClass *>(this)->visitModExpr(binOpExpr);
    }
    case ExprKind::FloorDiv: {
      auto binOpExpr = expr.cast<BinaryOpExpr>();
      return static_cast<SubClass *>(this)->visitFloorDivExpr(binOpExpr);
    }
    case ExprKind::CeilDiv: {
      auto binOpExpr = expr.cast<BinaryOpExpr>();
      return static_cast<SubClass *>(this)->visitCeilDivExpr(binOpExpr);
    }
    case ExprKind::Constant:
      return static_cast<SubClass *>(this)->visitConstantExpr(
          expr.cast<ConstantExpr>());
    case ExprKind::SymbolId:
      return static_cast<SubClass *>(this)->visitSymbolExpr(
          expr.cast<SymbolExpr>());
    }
    llvm_unreachable("Unknown Expr");
  }

  //===--------------------------------------------------------------------===//
  // Visitation functions... these functions provide default fallbacks in case
  // the user does not specify what to do for a particular instruction type.
  // The default behavior is to generalize the instruction type to its subtype
  // and try visiting the subtype.  All of this should be inlined perfectly,
  // because there are no virtual functions to get in the way.
  //

  // Default visit methods. Note that the default op-specific binary op visit
  // methods call the general visitBinaryOpExpr visit method.
  void visitBinaryOpExpr(BinaryOpExpr expr) {}
  void visitAddExpr(BinaryOpExpr expr) {
    static_cast<SubClass *>(this)->visitBinaryOpExpr(expr);
  }
  void visitMulExpr(BinaryOpExpr expr) {
    static_cast<SubClass *>(this)->visitBinaryOpExpr(expr);
  }
  void visitModExpr(BinaryOpExpr expr) {
    static_cast<SubClass *>(this)->visitBinaryOpExpr(expr);
  }
  void visitFloorDivExpr(BinaryOpExpr expr) {
    static_cast<SubClass *>(this)->visitBinaryOpExpr(expr);
  }
  void visitCeilDivExpr(BinaryOpExpr expr) {
    static_cast<SubClass *>(this)->visitBinaryOpExpr(expr);
  }
  void visitConstantExpr(ConstantExpr expr) {}
  void visitSymbolExpr(SymbolExpr expr) {}

private:
  // Walk the operands - each operand is itself walked in post order.
  void walkOperandsPostOrder(BinaryOpExpr expr) {
    walkPostOrder(expr.getLHS());
    walkPostOrder(expr.getRHS());
  }
};

class SimpleExprFlattener
    : public ExprVisitor<SimpleExprFlattener> {
public:
  // Flattend expression layout: [dims, symbols, locals, constant]
  // Stack that holds the LHS and RHS operands while visiting a binary op expr.
  // In future, consider adding a prepass to determine how big the SmallVector's
  // will be, and linearize this to std::vector<int64_t> to prevent
  // SmallVector moves on re-allocation.
  std::vector<SmallVector<int64_t, 8>> operandExprStack;

  unsigned numDims;
  unsigned numSymbols;

  // Number of newly introduced identifiers to flatten mod/floordiv/ceildiv's.
  unsigned numLocals;

  // Expr's corresponding to the floordiv/ceildiv/mod expressions for
  // which new identifiers were introduced; if the latter do not get canceled
  // out, these expressions can be readily used to reconstruct the Expr
  // (tree) form. Note that these expressions themselves would have been
  // simplified (recursively) by this pass. Eg. d0 + (d0 + 2*d1 + d0) ceildiv 4
  // will be simplified to d0 + q, where q = (d0 + d1) ceildiv 2. (d0 + d1)
  // ceildiv 2 would be the local expression stored for q.
  SmallVector<Expr, 4> localExprs;

  SimpleExprFlattener(unsigned numDims, unsigned numSymbols);

  virtual ~SimpleExprFlattener() = default;

  // Visitor method overrides.
  void visitMulExpr(BinaryOpExpr expr);
  void visitAddExpr(BinaryOpExpr expr);
  void visitSymbolExpr(SymbolExpr expr);
  void visitConstantExpr(ConstantExpr expr);
  void visitCeilDivExpr(BinaryOpExpr expr);
  void visitFloorDivExpr(BinaryOpExpr expr);

  //
  // t = expr mod c   <=>  t = expr - c*q and c*q <= expr <= c*q + c - 1
  //
  // A mod expression "expr mod c" is thus flattened by introducing a new local
  // variable q (= expr floordiv c), such that expr mod c is replaced with
  // 'expr - c * q' and c * q <= expr <= c * q + c - 1 are added to localVarCst.
  void visitModExpr(BinaryOpExpr expr);

protected:
  // Add a local identifier (needed to flatten a mod, floordiv, ceildiv expr).
  // The local identifier added is always a floordiv of a pure add/mul affine
  // function of other identifiers, coefficients of which are specified in
  // dividend and with respect to a positive constant divisor. localExpr is the
  // simplified tree expression (Expr) corresponding to the quantifier.
  virtual void addLocalFloorDivId(ArrayRef<int64_t> dividend, int64_t divisor,
                                  Expr localExpr);

private:
  // t = expr floordiv c   <=> t = q, c * q <= expr <= c * q + c - 1
  // A floordiv is thus flattened by introducing a new local variable q, and
  // replacing that expression with 'q' while adding the constraints
  // c * q <= expr <= c * q + c - 1 to localVarCst (done by
  // FlatConstraints::addLocalFloorDiv).
  //
  // A ceildiv is similarly flattened:
  // t = expr ceildiv c   <=> t =  (expr + c - 1) floordiv c
  void visitDivExpr(BinaryOpExpr expr, bool isCeil);

  int findLocalId(Expr localExpr);

  inline unsigned getNumCols() const {
    return numDims + numSymbols + numLocals + 1;
  }
  inline unsigned getConstantIndex() const { return getNumCols() - 1; }
  inline unsigned getLocalVarStartIndex() const { return numDims + numSymbols; }
  inline unsigned getSymbolStartIndex() const { return numDims; }
  inline unsigned getDimStartIndex() const { return 0; }
};
}

#endif // LLVM_EXPR_VISITOR_H
