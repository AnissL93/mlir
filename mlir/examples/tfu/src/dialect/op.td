#ifndef TFU_OPS
#define TFU_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Tfu_Dialect : Dialect {
  let name = "tfu";
  let cppNamespace = "tfu";
}

// Base class for tfu dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Tfu_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Tfu_Dialect, mnemonic, traits>;

// Provide a definition of the types that are used within the Tfu dialect.
def TfuType :
    Type<CPred<"$_self.isa<TfuType>()">, "Tfu type">;

//===----------------------------------------------------------------------===//
// Tfu Operations
//===----------------------------------------------------------------------===//

// We define a tfu operation by inheriting from our base 'Tfu_Op' class above.
// Here we provide the mnemonic and a list of traits for the operation. The
// constant operation is marked as 'NoSideEffect' as it is a pure operation
// and may be removed if dead.
def ConstantOp : Tfu_Op<"constant",
    [ConstantLike, NoSideEffect]> {
  let arguments = (ins F64Attr:$value, StrAttr:$op_name);
  let results = (outs TfuType);
}

def CopyOp : Tfu_Op<"copy", [NoSideEffect]> {
  let arguments = (ins TfuType:$src);
  let results = (outs TfuType:$dst);
}

def ConvOp : Tfu_Op<"conv", [NoSideEffect]> {
  let arguments = (ins TfuType:$input, TfuType:$weight, I64Attr:$stride, StrAttr:$op_name );
  let results = (outs TfuType:$output);
}

def ReturnOp : Tfu_Op<"return", [NoSideEffect, HasParent<"FuncOp">,
                                 Terminator]> {
  let summary = "return operation";
  let arguments = (ins Variadic<TfuType>:$input);

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];
}

#endif // TFU_OPS
