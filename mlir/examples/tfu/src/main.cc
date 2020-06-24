//
// Created by aniss on 2020/6/20.
//

#include <algorithm>
#include <iostream>
#include <numeric>

#include "dialect/dialect.h"

#include "llvm/Support/RandomNumberGenerator.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Attributes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace tfu {

struct Module {

  Module(MLIRContext* ctx, OpBuilder builder) : context(ctx), builder(builder) {
    I64 = builder.getIntegerType(64);
    F64 = builder.getF64Type();
    F16 = builder.getF16Type();
  }

  void MakeEntry(llvm::ArrayRef<Type> arg_types) {
    module = mlir::ModuleOp::create(builder.getUnknownLoc());

    auto func_type = builder.getFunctionType(arg_types, llvm::None);
    FuncOp entry_func = FuncOp::create(builder.getUnknownLoc(),
                                       "test-conv-conv", func_type);
    auto entry_block = entry_func.addEntryBlock();
    module.push_back(entry_func);
    builder.setInsertionPointToStart(entry_block);
  }

  Value ConstTensor(TensorType t) {
    auto init_attr = builder.getFloatAttr(F64, 0);
    return builder.create<ConstantOp>(builder.getUnknownLoc(), t,
                               init_attr);
  }

  Value ConstTensor(llvm::ArrayRef<int64_t> shape) {
    auto t_type = RankedTensorType::get(shape, F16);
    std::vector<double> data;
    for (size_t i = 0; i < shape.size(); ++i) {
      data.push_back(static_cast<double>(i));
    }
    auto data_attr = DenseElementsAttr::get(t_type, llvm::makeArrayRef(data));
    return builder.create<ConstantOp>(builder.getUnknownLoc(), t_type,
                               data_attr);
  }

  Value Conv(Value in, Value w, int s) {
    Type out_type = UnrankedTensorType::get(builder.getF16Type());
    return builder.create<ConvOp>(builder.getUnknownLoc(), out_type, in, w, builder.getIntegerAttr(I64, s));
  }

  void MakeFunc(llvm::StringRef name, Type ret_type,
                 llvm::ArrayRef<Type> args) {
    auto func_type = builder.getFunctionType(args, {ret_type});
    FuncOp func = FuncOp::create(builder.getUnknownLoc(), name, func_type);
    auto entry_block = func.addEntryBlock();
    builder.setInsertionPointToStart(entry_block);
  }

  void Return(Value in) {
    builder.create<ReturnOp>(builder.getUnknownLoc(), in);
  }

  void dump() {
    module.dump();
  }

  Type F64, I64, F16;
  ModuleOp module;
  OpBuilder builder;
  MLIRContext* context;
};

}
}

void buildConvConv() {
  using namespace mlir;
  using namespace mlir::tfu;
  MLIRContext context;
  OpBuilder builder(&context);
  Module module(&context, builder);

  auto in_type = RankedTensorType::get({1, 5, 5, 10}, builder.getF16Type());
  auto w_type = RankedTensorType::get({10, 1, 1, 10}, builder.getIntegerType(8));
  auto out_type = RankedTensorType::get({1, 5, 5, 10}, builder.getF16Type());
  llvm::SmallVector<mlir::Type, 3> arg_types = {in_type, w_type, out_type};
  module.MakeEntry(arg_types);
  Value input = module.ConstTensor(in_type);
  Value w1 = module.ConstTensor(w_type);
  Value w2 = module.ConstTensor(w_type);
  Value out = module.ConstTensor(out_type);
  Value conv1_out = module.Conv(input, w1, 1);
  Value conv2_out = module.Conv(conv1_out, w2, 1);
  module.Return(conv2_out);

  module.dump();
}


int main(int argn, char** argv) {
  mlir::registerAllDialects();
  mlir::registerDialect<mlir::tfu::TfuDialect>();
  buildConvConv();
  return 0;
}