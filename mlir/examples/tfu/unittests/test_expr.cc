//
// Created by aniss on 2020/6/21.
//

#include "src/dialect/type/expr.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;

TEST(AddExprTest, ExprTest) {
  MLIRContext context;
  Expr a = getConstantExpr(20, &context);
  Expr b = getConstantExpr(10, &context);
  Expr a_sym = getSymbolExpr(0, "a", &context);
  Expr c = a + a_sym;
  c.dump();
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}