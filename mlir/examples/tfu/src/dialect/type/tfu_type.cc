//
// Created by aniss on 2020/6/22.
//

#include "tfu_type.h"
#include "expr.h"
#include "mlir/IR/Types.h"
#include "ir/Expr.h"
#include "ir/Range.h"
#include "ir/IROperator.h"

#include <sstream>

using namespace HalideIR::Internal;

using Range = HalideIR::IR::Range;

namespace mlir {
namespace tfu {
namespace detail {

struct RegionStorage : public mlir::TypeStorage {

  using KeyTy = ::std::tuple<llvm::ArrayRef<Expr>, llvm::StringRef, Type>;

  RegionStorage(llvm::ArrayRef<Expr> shape, llvm::StringRef ms, Type type)
      : shape(shape), mem_scope(ms), element_type(type) {}

  bool operator==(const KeyTy &key) const {
    return ::std::get<0>(key) == shape && ::std::get<1>(key) == mem_scope &&
           ::std::get<2>(key) == element_type;
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(::std::get<0>(key), ::std::get<1>(key), ::std::get<2>(key));
  }

  static KeyTy getKey(llvm::ArrayRef<Expr> sh, llvm::StringRef ms, Type t) {
    return ::std::make_tuple(sh, ms, t);
  }

  static RegionStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    // Copy the elements from the provided `KeyTy` into the allocator.
    llvm::ArrayRef<Expr> shape =
        allocator.copyInto(::std::get<0>(key));

    llvm::StringRef name = allocator.copyInto(::std::get<1>(key));

    // Allocate the storage instance and construct it.
    return new (allocator.allocate<RegionStorage>())
        RegionStorage(shape, name, ::std::get<2>(key));
  }

  // shape
  llvm::ArrayRef<Expr> shape;
  llvm::StringRef mem_scope;
  Type element_type;
};

struct RangeTypeStorage : public mlir::TypeStorage {

  using KeyTy = Range;

  RangeTypeStorage(const Range & r) : range(r) {}

  bool operator==(const KeyTy &key) const {
    return key->min == range->min && key->extent == range->extent;
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.get());
  }

  static KeyTy getKey(const Range &r) {
    return r;
  }

  static RangeTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                  const KeyTy &key) {
    return new (allocator.allocate<RegionStorage>()) RangeTypeStorage(key);
  }

  Range range;
};
} // namespace detail

Region Region::get(llvm::ArrayRef<int64_t> shape, llvm::StringRef ms,
                   Type elem_type) {
  assert((shape.size() == 4) && "expected 4 dimensions in shape");

  mlir::MLIRContext* ctx = elem_type.getContext();

  // expr shape
  llvm::SmallVector<Expr, 4> shape_expr;
  for (size_t i = 0; i < shape.size(); ++i) {
    shape_expr.push_back(getConstantExpr(shape[i], ctx));
  }
  return Base::get(ctx,  TfuTypes::Region, shape_expr, ms, elem_type);
}

llvm::ArrayRef<Expr> Region::getShape() {
  return getImpl()->shape;
}

Type Region::getElemType() {
  return getImpl()->element_type;
}

llvm::StringRef Region::getMemScope() {
  return getImpl()->mem_scope;
}

void Region::print(llvm::raw_ostream &os) {
  os << "region<";
  for (int i = 0; i < getShape().size(); ++i) {
    getShape()[i].print(os);
    os << "x";
  }
  os << getMemScope() << "x" << getElemType() << ">";
}

void Region::dump() {
  print(llvm::errs());
}

////////////////////// range type
RangeType RangeType::get(int64_t st, int64_t ed, MLIRContext* ctx) {
  Range r =
      HalideIR::IR::Range::make_by_min_extent(
          make_const(::HalideIR::Int(64), st),
          make_const(::HalideIR::Int(64), ed - st));

  return Base::get(ctx,  RangeTypes::Type, r);
}

HalideIR::Expr RangeType::getStart() {
  return getImpl()->range->min;
}

HalideIR::Expr RangeType::getExtent() {
  return getImpl()->range->extent;
}

void RangeType::print(llvm::raw_ostream& os) {
  ::std::ostringstream str;
  using namespace HalideIR;
  str << "range<" << getStart() << ", " << getExtent() << ">";
  os << str.str();
}

void RangeType::dump() {
  print(llvm::errs());
}

} // namespace tfu
} // namespace mlir