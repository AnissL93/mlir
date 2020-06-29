//
// Created by aniss on 2020/6/22.
//

#ifndef LLVM_TFU_TYPE_H
#define LLVM_TFU_TYPE_H

#include "dialect/dialect.h"
//#include "expr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/raw_os_ostream.h"
#include "ir/Expr.h"

namespace mlir {
namespace tfu {

namespace detail {
struct TfuTypeStorage;
}

namespace TfuTypes {
enum Types {
  TfuType = mlir::Type::FIRST_TFU_TYPE,
};
} // end namespace ToyTypes

class TfuType : public mlir::Type::TypeBase<TfuType, mlir::Type,
    detail::TfuTypeStorage> {
public:
  using Base::Base;
  static bool kindof(unsigned kind) { return kind == TfuTypes::Types::TfuType; }

  static TfuType get(llvm::ArrayRef<int64_t> shape, llvm::StringRef ms, Type elem_type);

  llvm::ArrayRef<HExpr> getShape();

  llvm::StringRef getMemScope();

  Type getElemType();

  void print(llvm::raw_ostream& os);

  void dump();
};

namespace detail {
struct RangeTypeStorage;
}

namespace RangeTypes {
enum Types {
  Type = mlir::Type::FIRST_RANGE_TYPE,
};
} // end namespace ToyTypes

class RangeType : public mlir::Type::TypeBase<RangeType, mlir::Type,
    detail::RangeTypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == RangeTypes::Type; }

  static RangeType get(int64_t st, int64_t ed, ::mlir::MLIRContext* ctx);

  HExpr getStart();

  HExpr getExtent();

  void print(llvm::raw_ostream& os);

  void dump();
};

} // tfu
} // mlir

#endif // LLVM_TFU_TYPE_H
