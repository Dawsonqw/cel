#ifndef DIALECT_IR_CELOPS_H
#define DIALECT_IR_CELOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Dialect/IR/CelDialect.h.inc"
#define GET_OP_CLASSES
#include "Dialect/IR/CelOps.h.inc"

using namespace mlir;

#endif