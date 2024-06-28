#ifndef UTILS_COMMON_H
#define UTILS_COMON_H
#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/ADT/StringRef.h"

#include <glog/logging.h>

namespace mlir
{
    namespace cel
    {
        mlir::NameLoc getNameLoc(mlir::Builder builder, llvm::StringRef info);

        void loadMLIR(const std::string &inputFilename, mlir::MLIRContext &context,
                  mlir::OwningOpRef<mlir::ModuleOp> &module);

        void RegisterDialect(mlir::MLIRContext &context,const mlir::DialectRegistry& registry);
    }
}

#endif