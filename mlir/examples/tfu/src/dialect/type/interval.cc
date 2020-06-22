//
// Created by aniss on 2020/6/21.
//

#include "interval.h"
#include "interval_detail.h"

using namespace mlir;
using namespace mlir::detail;

MLIRContext * Interval::getContext() const {
  return impl->context;
}

IntervalKind Interval::getKind() const {
  return static_cast<IntervalKind>(impl->getKind());
}

template<typename U>
bool Interval::isa() const {
  if (std::is_same<U, StaticInterval>::value) {
    return true;
  } else if (std::is_same<U, DynamicInterval>::value) {
    return true;
  } else {
    return false;
  }
}

template<typename U>
U Interval::cast() const {

}

