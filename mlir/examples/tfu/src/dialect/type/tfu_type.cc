//
// Created by aniss on 2020/6/22.
//

#include "tfu_type.h"
//#include "expr.h"
#include "mlir/IR/Types.h"
#include "ir/Expr.h"
#include "ir/Range.h"
#include "ir/IROperator.h"

#include "base/Type.h"

#include <sstream>

namespace mlir {
namespace tfu {
namespace detail {

struct RegionStorage : public mlir::TypeStorage {

  using KeyTy = ::std::tuple<llvm::ArrayRef<HExpr>, llvm::StringRef, Type>;

  RegionStorage(llvm::ArrayRef<HExpr> shape, llvm::StringRef ms, Type type)
      : shape(shape), mem_scope(ms), element_type(type) {}

  bool operator==(const KeyTy &key) const {
    for (int i = 0; i < shape.size(); ++i) {
      if (std::get<0>(key)[i].get() != shape[i].get()) {
        return false;
      }
    }
    return ::std::get<1>(key) == mem_scope &&
           ::std::get<2>(key) == element_type;
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    std::vector<const void*> shape_pointers;
    auto shape = std::get<0>(key);
    llvm::hash_code code = llvm::hash_combine(shape[0].get());
    for (int i = 1; i < shape.size(); ++i) {
      code = llvm::hash_combine(code, shape[i].get());
    }
    return llvm::hash_combine(code, ::std::get<1>(key), ::std::get<2>(key));
  }

  static KeyTy getKey(llvm::ArrayRef<HExpr> sh, llvm::StringRef ms, Type t) {
    return std::make_tuple(sh, ms, t);
  }

  static RegionStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    // Copy the elements from the provided `KeyTy` into the allocator.
    llvm::ArrayRef<HExpr> shape =
        allocator.copyInto(::std::get<0>(key));

    llvm::StringRef name = allocator.copyInto(::std::get<1>(key));

    // Allocate the storage instance and construct it.
    return new (allocator.allocate<RegionStorage>())
        RegionStorage(shape, name, ::std::get<2>(key));
  }

  // shape
  llvm::ArrayRef<HExpr> shape;
  llvm::StringRef mem_scope;
  Type element_type;
};

struct RangeTypeStorage : public mlir::TypeStorage {

  using KeyTy = HalideIR::IR::Range;

  RangeTypeStorage(const HalideIR::IR::Range & r) : range(r) {}

  bool operator==(const KeyTy &key) const {
    return range.get() == key.get();
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.get());
  }

  static KeyTy getKey(const HalideIR::IR::Range &r) {
    return r;
  }

  static RangeTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                  const KeyTy &key) {
    return new (allocator.allocate<RegionStorage>()) RangeTypeStorage(key);
  }

  HalideIR::IR::Range range;
};
} // namespace detail

Region Region::get(llvm::ArrayRef<int64_t> shape, llvm::StringRef ms,
                   mlir::Type elem_type) {
  assert((shape.size() == 4) && "expected 4 dimensions in shape");

  mlir::MLIRContext* ctx = elem_type.getContext();

  // expr shape
  llvm::SmallVector<HExpr, 4> shape_expr;
  for (size_t i = 0; i < shape.size(); ++i) {
    shape_expr.push_back(HalideIR::Internal::make_const(HalideIR::Int(64), shape[i]));
  }
  return Base::get(ctx,  TfuTypes::Region, shape_expr, ms, elem_type);
}

llvm::ArrayRef<HExpr> Region::getShape() {
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
    std::ostringstream str;
    str << getShape()[i];
    os << str.str();
    os << "x";
  }
  os << getMemScope() << "x" << getElemType() << ">";
}

void Region::dump() {
  print(llvm::errs());
}


} // namespace tfu
} // namespace mlir

////////////////////// range type
mlir::tfu::RangeType mlir::tfu::RangeType::get(int64_t st, int64_t ed, mlir::MLIRContext* ctx) {
      HalideIR::IR::Range r =
      HalideIR::IR::Range::make_by_min_extent(
          HalideIR::Internal::make_const(::HalideIR::Int(64), st),
          HalideIR::Internal::make_const(::HalideIR::Int(64), ed - st));

  return Base::get(ctx, RangeTypes::Type, r);
}

HExpr mlir::tfu::RangeType::getStart() {
  return getImpl()->range->min;
}

HExpr mlir::tfu::RangeType::getExtent() {
  return getImpl()->range->extent;
}

void mlir::tfu::RangeType::print(llvm::raw_ostream& os) {
  ::std::ostringstream str;
  using namespace HalideIR;
  str << "range<" << getStart() << ", " << getExtent() << ">";
  os << str.str();
}

void mlir::tfu::RangeType::dump() {
  print(llvm::errs());
}
