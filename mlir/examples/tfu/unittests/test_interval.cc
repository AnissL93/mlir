//
// Created by aniss on 2020/6/27.
//

#include "dialect/type/interval.h"

int main() {
  using namespace tfu;
  VarExpr var("i");
  Iteration iter = Iteration::makeByFactor(var, 55, 10);
  std::cout << iter << std::endl;

  VarExpr size("size");
  HExpr factor = HalideIR::Internal::make_const(HalideIR::Int(64), 10);
  Iteration dyn_iter = Iteration::makeByFactor(var, size, factor);
  std::cout << dyn_iter << std::endl;
  Iteration reuse = ComputeReuse(dyn_iter);
  std::cout << reuse << std::endl;
  auto range0 = dyn_iter.GetRangeOf(2);
  std::cout << range0.min << ", " << range0.max << std::endl;
  return 0;
}


