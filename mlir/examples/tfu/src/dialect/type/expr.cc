//
// Created by aniss on 2020/6/20.
//

#include "dialect/type/expr.h"
#include "dialect/type/expr_detail.h"
#include "expr_visitor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_os_ostream.h"

using namespace mlir;
using namespace mlir::detail;

MLIRContext *Expr::getContext() const { return expr->context; }

ExprKind Expr::getKind() const {
  return static_cast<ExprKind>(expr->getKind());
}

/// Walk all of the Exprs in this subgraph in postorder.
void Expr::walk(std::function<void(Expr)> callback) const {
  struct ExprWalker : public ExprVisitor<ExprWalker> {
    std::function<void(Expr)> callback;

    ExprWalker(std::function<void(Expr)> callback) : callback(callback) {}

    void visitBinaryOpExpr(BinaryOpExpr expr) { callback(expr); }
    void visitConstantExpr(ConstantExpr expr) { callback(expr); }
    void visitSymbolExpr(SymbolExpr expr) { callback(expr); }
  };

  ExprWalker(callback).walkPostOrder(*this);
}

// Dispatch affine expression construction based on kind.
Expr mlir::getBinaryOpExpr(ExprKind kind, Expr lhs, Expr rhs) {
  if (kind == ExprKind::Add)
    return lhs + rhs;
  if (kind == ExprKind::Mul)
    return lhs * rhs;
  if (kind == ExprKind::FloorDiv)
    return lhs.floorDiv(rhs);
  if (kind == ExprKind::CeilDiv)
    return lhs.ceilDiv(rhs);
  if (kind == ExprKind::Mod)
    return lhs % rhs;

  llvm_unreachable("unknown binary operation on affine expressions");
}

/// Returns true if this expression is made out of only symbols and
/// constants (no dimensional identifiers).
bool Expr::isSymbolicOrConstant() const {
  switch (getKind()) {
  case ExprKind::Constant:
    return true;
  case ExprKind::SymbolId:
    return true;

  case ExprKind::Add:
  case ExprKind::Mul:
  case ExprKind::FloorDiv:
  case ExprKind::CeilDiv:
  case ExprKind::Max:
  case ExprKind::Min:
  case ExprKind::Sub:
  case ExprKind::Mod: {
    auto expr = this->cast<BinaryOpExpr>();
    return expr.getLHS().isSymbolicOrConstant() &&
           expr.getRHS().isSymbolicOrConstant();
  }
  }
  llvm_unreachable("Unknown Expr");
}

/// Returns true if this is a pure affine expression, i.e., multiplication,
/// floordiv, ceildiv, and mod is only allowed w.r.t constants.
bool Expr::isPure() const {
  switch (getKind()) {
  case ExprKind::SymbolId:
  case ExprKind::Constant:
    return true;
  case ExprKind::Max:
  case ExprKind::Min:
  case ExprKind::Sub:
  case ExprKind::Add: {
    auto op = cast<BinaryOpExpr>();
    return op.getLHS().isPure() && op.getRHS().isPure();
  }
  case ExprKind::Mul: {
    // TODO: Canonicalize the constants in binary operators to the RHS when
    // possible, allowing this to merge into the next case.
    auto op = cast<BinaryOpExpr>();
    return op.getLHS().isPure() && op.getRHS().isPure() &&
           (op.getLHS().template isa<ConstantExpr>() ||
            op.getRHS().template isa<ConstantExpr>());
  }
  case ExprKind::FloorDiv:
  case ExprKind::CeilDiv:
  case ExprKind::Mod: {
    auto op = cast<BinaryOpExpr>();
    return op.getLHS().isPure() && op.getRHS().template isa<ConstantExpr>();
  }
  }
  llvm_unreachable("Unknown Expr");
}

BinaryOpExpr::BinaryOpExpr(Expr::ImplType *ptr) : Expr(ptr) {}
Expr BinaryOpExpr::getLHS() const { return static_cast<ImplType *>(expr)->lhs; }
Expr BinaryOpExpr::getRHS() const { return static_cast<ImplType *>(expr)->rhs; }

SymbolExpr::SymbolExpr(Expr::ImplType *ptr) : Expr(ptr) {}
unsigned SymbolExpr::getPosition() const {
  return static_cast<ImplType *>(expr)->position;
}

ConstantExpr::ConstantExpr(Expr::ImplType *ptr) : Expr(ptr) {}
int64_t ConstantExpr::getValue() const {
  return static_cast<ImplType *>(expr)->constant;
}

bool Expr::operator==(int64_t v) const {
  return *this == getConstantExpr(v, getContext());
}

Expr mlir::getSymbolExpr(unsigned int position, llvm::StringRef name,
                         MLIRContext *context) {
  auto assignCtx = [context](SymbolExprStorage *storage) {
    storage->context = context;
  };

  StorageUniquer &uniquer = context->getExprUniquer();
  return uniquer.get<SymbolExprStorage>(
      assignCtx, static_cast<unsigned>(ExprKind::SymbolId), position, name);
}

Expr mlir::getConstantExpr(int64_t constant, MLIRContext *context) {
  auto assignCtx = [context](ConstantExprStorage *storage) {
    storage->context = context;
  };

  StorageUniquer &uniquer = context->getExprUniquer();
  return uniquer.get<ConstantExprStorage>(
      assignCtx, static_cast<unsigned>(ExprKind::Constant), constant);
}

/// Simplify add expression. Return nullptr if it can't be simplified.
static Expr simplifyAdd(Expr lhs, Expr rhs) {
  auto lhsConst = lhs.dyn_cast<ConstantExpr>();
  auto rhsConst = rhs.dyn_cast<ConstantExpr>();
  // Fold if both LHS, RHS are a constant.
  if (lhsConst && rhsConst)
    return getConstantExpr(lhsConst.getValue() + rhsConst.getValue(),
                           lhs.getContext());

  // Canonicalize so that only the RHS is a constant. (4 + d0 becomes d0 + 4).
  // If only one of them is a symbolic expressions, make it the RHS.
  if (lhs.isa<ConstantExpr>() ||
      (lhs.isSymbolicOrConstant() && !rhs.isSymbolicOrConstant())) {
    return rhs + lhs;
  }

  // At this point, if there was a constant, it would be on the right.

  // Addition with a zero is a noop, return the other input.
  if (rhsConst) {
    if (rhsConst.getValue() == 0)
      return lhs;
  }
  // Fold successive additions like (d0 + 2) + 3 into d0 + 5.
  auto lBin = lhs.dyn_cast<BinaryOpExpr>();
  if (lBin && rhsConst && lBin.getKind() == ExprKind::Add) {
    if (auto lrhs = lBin.getRHS().dyn_cast<ConstantExpr>())
      return lBin.getLHS() + (lrhs.getValue() + rhsConst.getValue());
  }

  // Detect "c1 * expr + c_2 * expr" as "(c1 + c2) * expr".
  // c1 is rRhsConst, c2 is rLhsConst; firstExpr, secondExpr are their
  // respective multiplicands.
  Optional<int64_t> rLhsConst, rRhsConst;
  Expr firstExpr, secondExpr;
  ConstantExpr rLhsConstExpr;
  auto lBinOpExpr = lhs.dyn_cast<BinaryOpExpr>();
  if (lBinOpExpr && lBinOpExpr.getKind() == ExprKind::Mul &&
      (rLhsConstExpr = lBinOpExpr.getRHS().dyn_cast<ConstantExpr>())) {
    rLhsConst = rLhsConstExpr.getValue();
    firstExpr = lBinOpExpr.getLHS();
  } else {
    rLhsConst = 1;
    firstExpr = lhs;
  }

  auto rBinOpExpr = rhs.dyn_cast<BinaryOpExpr>();
  ConstantExpr rRhsConstExpr;
  if (rBinOpExpr && rBinOpExpr.getKind() == ExprKind::Mul &&
      (rRhsConstExpr = rBinOpExpr.getRHS().dyn_cast<ConstantExpr>())) {
    rRhsConst = rRhsConstExpr.getValue();
    secondExpr = rBinOpExpr.getLHS();
  } else {
    rRhsConst = 1;
    secondExpr = rhs;
  }

  if (rLhsConst && rRhsConst && firstExpr == secondExpr)
    return getBinaryOpExpr(
        ExprKind::Mul, firstExpr,
        getConstantExpr(rLhsConst.getValue() + rRhsConst.getValue(),
                        lhs.getContext()));

  // When doing successive additions, bring constant to the right: turn (d0 + 2)
  // + d1 into (d0 + d1) + 2.
  if (lBin && lBin.getKind() == ExprKind::Add) {
    if (auto lrhs = lBin.getRHS().dyn_cast<ConstantExpr>()) {
      return lBin.getLHS() + rhs + lrhs;
    }
  }

  // Detect and transform "expr - c * (expr floordiv c)" to "expr mod c". This
  // leads to a much more efficient form when 'c' is a power of two, and in
  // general a more compact and readable form.

  // Process '(expr floordiv c) * (-c)'.
  if (!rBinOpExpr)
    return nullptr;

  auto lrhs = rBinOpExpr.getLHS();
  auto rrhs = rBinOpExpr.getRHS();

  // Process lrhs, which is 'expr floordiv c'.
  BinaryOpExpr lrBinOpExpr = lrhs.dyn_cast<BinaryOpExpr>();
  if (!lrBinOpExpr || lrBinOpExpr.getKind() != ExprKind::FloorDiv)
    return nullptr;

  auto llrhs = lrBinOpExpr.getLHS();
  auto rlrhs = lrBinOpExpr.getRHS();

  if (lhs == llrhs && rlrhs == -rrhs) {
    return lhs % rlrhs;
  }
  return nullptr;
}

Expr Expr::operator+(int64_t v) const {
  return *this + getConstantExpr(v, getContext());
}
Expr Expr::operator+(Expr other) const {
  if (auto simplified = simplifyAdd(*this, other))
    return simplified;

  StorageUniquer &uniquer = getContext()->getExprUniquer();
  return uniquer.get<BinaryOpExprStorage>(
      /*initFn=*/{}, static_cast<unsigned>(ExprKind::Add), *this, other);
}

/// Simplify a multiply expression. Return nullptr if it can't be simplified.
static Expr simplifyMul(Expr lhs, Expr rhs) {
  auto lhsConst = lhs.dyn_cast<ConstantExpr>();
  auto rhsConst = rhs.dyn_cast<ConstantExpr>();

  if (lhsConst && rhsConst)
    return getConstantExpr(lhsConst.getValue() * rhsConst.getValue(),
                           lhs.getContext());

  assert(lhs.isSymbolicOrConstant() || rhs.isSymbolicOrConstant());

  // Canonicalize the mul expression so that the constant/symbolic term is the
  // RHS. If both the lhs and rhs are symbolic, swap them if the lhs is a
  // constant. (Note that a constant is trivially symbolic).
  if (!rhs.isSymbolicOrConstant() || lhs.isa<ConstantExpr>()) {
    // At least one of them has to be symbolic.
    return rhs * lhs;
  }

  // At this point, if there was a constant, it would be on the right.

  // Multiplication with a one is a noop, return the other input.
  if (rhsConst) {
    if (rhsConst.getValue() == 1)
      return lhs;
    // Multiplication with zero.
    if (rhsConst.getValue() == 0)
      return rhsConst;
  }

  // Fold successive multiplications: eg: (d0 * 2) * 3 into d0 * 6.
  auto lBin = lhs.dyn_cast<BinaryOpExpr>();
  if (lBin && rhsConst && lBin.getKind() == ExprKind::Mul) {
    if (auto lrhs = lBin.getRHS().dyn_cast<ConstantExpr>())
      return lBin.getLHS() * (lrhs.getValue() * rhsConst.getValue());
  }

  // When doing successive multiplication, bring constant to the right: turn (d0
  // * 2) * d1 into (d0 * d1) * 2.
  if (lBin && lBin.getKind() == ExprKind::Mul) {
    if (auto lrhs = lBin.getRHS().dyn_cast<ConstantExpr>()) {
      return (lBin.getLHS() * rhs) * lrhs;
    }
  }

  return nullptr;
}

Expr Expr::operator*(int64_t v) const {
  return *this * getConstantExpr(v, getContext());
}
Expr Expr::operator*(Expr other) const {
  if (auto simplified = simplifyMul(*this, other))
    return simplified;

  StorageUniquer &uniquer = getContext()->getExprUniquer();
  return uniquer.get<BinaryOpExprStorage>(
      /*initFn=*/{}, static_cast<unsigned>(ExprKind::Mul), *this, other);
}

// Unary minus, delegate to operator*.
Expr Expr::operator-() const {
  return *this * getConstantExpr(-1, getContext());
}

// Delegate to operator+.
Expr Expr::operator-(int64_t v) const { return *this + (-v); }
Expr Expr::operator-(Expr other) const { return *this + (-other); }

Expr Expr::floorDiv(uint64_t v) const {
  return floorDiv(getConstantExpr(v, getContext()));
}
Expr Expr::floorDiv(Expr other) const {
  StorageUniquer &uniquer = getContext()->getExprUniquer();
  return uniquer.get<BinaryOpExprStorage>(
      /*initFn=*/{}, static_cast<unsigned>(ExprKind::FloorDiv), *this, other);
}

static Expr simplifyCeilDiv(Expr lhs, Expr rhs) {
  auto lhsConst = lhs.dyn_cast<ConstantExpr>();
  auto rhsConst = rhs.dyn_cast<ConstantExpr>();

  if (!rhsConst || rhsConst.getValue() < 1)
    return nullptr;

  if (lhsConst)
    return getConstantExpr(ceilDiv(lhsConst.getValue(), rhsConst.getValue()),
                           lhs.getContext());

  // Fold ceildiv of a multiply with a constant that is a multiple of the
  // divisor. Eg: (i * 128) ceildiv 64 = i * 2.
  if (rhsConst.getValue() == 1)
    return lhs;

  // Simplify (expr * const) ceildiv divConst when const is known to be a
  // multiple of divConst.
  auto lBin = lhs.dyn_cast<BinaryOpExpr>();
  if (lBin && lBin.getKind() == ExprKind::Mul) {
    if (auto lrhs = lBin.getRHS().dyn_cast<ConstantExpr>()) {
      // rhsConst is known to be a positive constant.
      if (lrhs.getValue() % rhsConst.getValue() == 0)
        return lBin.getLHS() * (lrhs.getValue() / rhsConst.getValue());
    }
  }

  return nullptr;
}

Expr Expr::ceilDiv(uint64_t v) const {
  return ceilDiv(getConstantExpr(v, getContext()));
}
Expr Expr::ceilDiv(Expr other) const {
  if (auto simplified = simplifyCeilDiv(*this, other))
    return simplified;

  StorageUniquer &uniquer = getContext()->getExprUniquer();
  return uniquer.get<BinaryOpExprStorage>(
      /*initFn=*/{}, static_cast<unsigned>(ExprKind::CeilDiv), *this, other);
}

Expr Expr::operator%(uint64_t v) const {
  return *this % getConstantExpr(v, getContext());
}
Expr Expr::operator%(Expr other) const {
  StorageUniquer &uniquer = getContext()->getExprUniquer();
  return uniquer.get<BinaryOpExprStorage>(
      /*initFn=*/{}, static_cast<unsigned>(ExprKind::Mod), *this, other);
}

void Expr::print(raw_ostream &os) const {
  ExprKind kind = getKind();
  if (BinaryOpExpr bin_op = dyn_cast<BinaryOpExpr>()) {
    Expr lhs = bin_op.getLHS();
    Expr rhs = bin_op.getRHS();
    os << "(";
    lhs.print(os);
    os << " ";
    // print +, -, *
    switch (kind) {
    case ExprKind::Add:
      os << "+";
      break;
    case ExprKind::Mul:
      os << "*";
      break;
    case ExprKind::Sub:
      os << "-";
      break;
    case ExprKind::CeilDiv:
      os << "ceil/";
      break;
    case ExprKind::FloorDiv:
      os << "floor/";
      break;
    case ExprKind::Max:
      os << "max";
      break;
    case ExprKind::Min:
      os << "min";
      break;
    default:
      break;
    }
    os << " ";
    rhs.print(os);
    os << ")";
    return;
  }

  if (ConstantExpr const_expr = dyn_cast<ConstantExpr>()) {
    os << const_expr.getValue();
    return;
  }

  if (SymbolExpr symbol_expr = dyn_cast<SymbolExpr>()) {
    os << "(" << symbol_expr.getNameHint() << ", " << symbol_expr.getPosition()
       << ")";
    return;
  }
}

void Expr::dump() const {
  print(llvm::errs());
}

raw_ostream &mlir::operator<<(raw_ostream &os, Expr expr) {
  expr.print(os);
  return os;
}

llvm::StringRef SymbolExpr::getNameHint() const {
  return static_cast<ImplType *>(expr)->name_hint;
}
