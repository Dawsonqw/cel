#include <cstdlib>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include <string>
#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

#include "Dialect/IR/CelOps.hpp"
#include "Utils/common.hpp"

int main(int argc, char **argv)
{
    llvm::cl::opt<std::string> inputFilename("mlir-input",
                                             llvm::cl::desc("<mlir file path>"),
                                             llvm::cl::init("default.mlir"),
                                             llvm::cl::value_desc("filename"));
    
     llvm::cl::opt<std::string> outputFilename("mlir-output",
                                              llvm::cl::desc("<mlir file path>"),
                                              llvm::cl::init("default_save.mlir"),
                                              llvm::cl::value_desc("filename"));
    
    llvm::cl::ParseCommandLineOptions(argc, argv);
    const std::string &inputFilenameStr = inputFilename.getValue();
    const std::string &outputFilenameStr = outputFilename.getValue();

    FLAGS_stderrthreshold = 1;
    FLAGS_minloglevel = google::ERROR;
    FLAGS_log_dir="./log";
    google::InitGoogleLogging(argv[0]);

    mlir::MLIRContext context;
    mlir::DialectRegistry registry;
    registry.insert<mlir::cel::CelDialect>();
    mlir::cel::RegisterDialect(context,registry);
    mlir::OwningOpRef<mlir::ModuleOp> module;
    mlir::cel::loadMLIR(inputFilenameStr, context, module);
    std::error_code ec;
    llvm::raw_fd_ostream output_file(outputFilenameStr, ec);
    module->print(output_file, mlir::OpPrintingFlags().useLocalScope().enableDebugInfo());
    return 0;
}