//
// Created by aniss on 2020/6/22.
//

#ifndef LLVM_TFU_TYPE_DETAIL_H
#define LLVM_TFU_TYPE_DETAIL_H

namespace mlir {
namespace tfu {

namespace detail {
struct RegionStorage;
}

namespace TfuTypes {
enum Types {
  Region = mlir::Type::FIRST_TFU_TYPE,
};
} // end namespace ToyTypes

class Region : public mlir::Type::TypeBase<Region, mlir::Type,
    detail::RegionStorage> {
public:
  using Base::Base;
  static bool kindof(unsigned kind) { return kind == TfuTypes::Types::Region; }

  static Region get(llvm::ArrayRef<unsigned> shape);
};

}

}

#endif // LLVM_TFU_TYPE_DETAIL_H
