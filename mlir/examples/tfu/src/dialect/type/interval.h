//
// Created by aniss on 2020/6/21.
//

#ifndef LLVM_SRC_DIALECT_TYPE_INTERVAL_H
#define LLVM_SRC_DIALECT_TYPE_INTERVAL_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/Support/Casting.h"
#include <type_traits>

namespace mlir {

class MLIRContext;

enum class IntervalKind {
  kDynamic, kStatic
};

namespace detail {
struct IntervalStorage;
struct DynamicIntervalStorage;
struct StaticIntervalStorage;
} // namespace detail

class Interval {
public:
  using ImplType = detail::IntervalStorage;
  bool operator==(Interval other) const { return impl == other.impl; }
  bool operator!=(Interval other) const { return !(*this == other); }
  bool operator==(int64_t v) const;
  bool operator!=(int64_t v) const { return !(*this == v); }
  explicit operator bool() const { return impl; }
  bool operator!() const { return impl == nullptr; }
  template <typename U> bool isa() const;
  template <typename U> U dyn_cast() const;
  template <typename U> U dyn_cast_or_null() const;
  template <typename U> U cast() const;
  IntervalKind getKind() const;
  MLIRContext *getContext() const;
  friend ::llvm::hash_code hash_value(Interval arg);
protected:
  ImplType * impl;
};

struct DynamicInterval : public Interval {
  using ImplType  = detail::DynamicIntervalStorage;
};

struct StaticInterval : public Interval {
  using ImplType  = detail::StaticIntervalStorage;
};

} // namespace mlir

#endif // LLVM_SRC_DIALECT_TYPE_INTERVAL_H
