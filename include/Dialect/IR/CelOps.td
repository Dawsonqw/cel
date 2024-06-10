#ifndef CEL_DIALECT_OPS_TD
#define CEL_DIALECT_OPS_TD
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "CelType.td"

def Cel_Dialect : Dialect {
    let name = "Cel";
    let summary = "Cel Dialect";
    let cppNamespace = "::mlir::cel";
    let dependentDialects = ["func::FuncDialect"];
}

class Cel_Op<string mnemonic, list<Trait> traits = [Pure]> : Op<Cel_Dialect, mnemonic, traits>;

def FuncOp : Cel_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  let summary = "user defined function operation";
  let description = [{
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }
  }];
}

def ReturnOp : Cel_Op<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {
  let summary = "return operation";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$input);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  // let hasVerifier = 1;
}

def CallOp : Cel_Op<"call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "generic call operation";
  let description = [{
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$inputs);

  let results = (outs AnyType: $Cel_Type);

  // let builders = [
  //   OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
  // ];
}

def DataOp: Cel_Op<"Data"> {
    let summary = "Data operation";
    let description = [{
    }];
    let arguments = (ins 
        AnyAttr:$value
    );
    let results = (outs AnyRankedTensor:$output);
}

#endif //