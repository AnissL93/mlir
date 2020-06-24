//
// Created by cambricon on 2020/6/24.
//

#include "base/Type.h"
#include "ir/Expr.h"
#include "ir/IROperator.h"

using HalideIR::Expr;
using namespace HalideIR::Internal;

int main() {

  HalideIR::Type t = HalideIR::Int(64);
  Expr a = make_const(t, 10);
  Expr b = make_const(t, 20);

  Expr c = a + b;

  std::cout << c << std::endl;

  Range r = Range::make_by_min_extent(a, b);
  std::cout << r << std::endl;
  return 0;
}