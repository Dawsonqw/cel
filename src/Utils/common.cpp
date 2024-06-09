#include "Utils/common.hpp"

namespace mlir
{
    namespace cel
    {
        mlir::NameLoc getNameLoc(mlir::Builder builder, llvm::StringRef info){
            return mlir::NameLoc::get(builder.getStringAttr(info));
        }


        void loadMLIR(const std::string &inputFilename, mlir::MLIRContext &context,
                    mlir::OwningOpRef<mlir::ModuleOp> &module)
        {
            std::string errorMessage;
            auto input = mlir::openInputFile(inputFilename, &errorMessage);
            if (!input)
            {
                LOG(FATAL) << "Error can't open file " << inputFilename << ". The error message is: " << errorMessage;
            }
            llvm::SourceMgr sourceMgr;
            mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
            sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
            module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
            if (!module)
            {
                LOG(FATAL) << "Source mlir file exists error,please check it";
            }
        }

        void RegisterDialect(mlir::MLIRContext &context,const mlir::DialectRegistry& registry)
        {
            context.appendDialectRegistry(registry);
            context.loadAllAvailableDialects();
        }
    }
}