#include "mlir/InitAllDialects.h"

#include "Dialect/IR/CelOps.hpp"

using namespace mlir;
using namespace mlir::cel;

#include "Dialect/IR/CelDialect.cpp.inc"

void CelDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "Dialect/IR/CelOps.cpp.inc"
        >();
}

#define GET_OP_CLASSES
#include "Dialect/IR/CelOps.cpp.inc"