//
// Created by aniss on 2020/6/21.
//

#ifndef LLVM_INTERVAL_DETAIL_H
#define LLVM_INTERVAL_DETAIL_H

#include "interval.h"
#include "expr.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/StorageUniquer.h"

namespace mlir {

class MLIRContext;

namespace detail {

struct IntervalStorage : public StorageUniquer::BaseStorage {
  MLIRContext* context;
};

struct StaticIntervalStorage : public IntervalStorage {
  // lb, ub, stride, extent
  using PatternTy = std::tuple<unsigned, unsigned, unsigned, unsigned>;
  using KeyTy = llvm::ArrayRef<PatternTy>;

  bool operator==(const KeyTy& key) const {
    for (size_t i = 0; i < key.size(); ++i) {
      const PatternTy &p = key[i];
      if (std::get<0>(p) != std::get<0>(value[i])) {
        return false;
      }
      if (std::get<1>(p) != std::get<1>(value[i])) {
        return false;
      }
      if (std::get<2>(p) != std::get<2>(value[i])) {
        return false;
      }
      if (std::get<3>(p) != std::get<3>(value[i])) {
        return false;
      }
    }
    return true;
  }

  static StaticIntervalStorage*
  construct(StorageUniquer::StorageAllocator &allocator,
            const KeyTy& key) {
    auto * result = allocator.allocate<StaticIntervalStorage>();
    result->value = key;
    return result;
  }

  KeyTy value;
};

struct DynamicIntervalStorage : public IntervalStorage {
  using KeyTy = std::tuple<Expr, Expr, Expr>;

  bool operator==(const KeyTy& key) const {
    return std::get<0>(key) == var
        && std::get<1>(key) == min
           && std::get<2>(key) == extent;
  }

  static DynamicIntervalStorage*
      construct(StorageUniquer::StorageAllocator &allocator,
                const KeyTy& key) {
    auto * result = allocator.allocate<DynamicIntervalStorage>();
    result->var = std::get<0>(key);
    result->min = std::get<1>(key);
    result->extent = std::get<2>(key);
    return result;
  }

  Expr var, min, extent;
};

} // namespace detail
} // namespace mlir

#endif // LLVM_INTERVAL_DETAIL_H
