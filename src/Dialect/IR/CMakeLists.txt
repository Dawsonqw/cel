file(GLOB _sources
          CelOps.cpp
          CelDialect.cpp
    )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_mlir_library(CelBasic
    ${_sources}
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include

    DEPENDS
    cel_dialect_gen

    LINK_LIBS PRIVATE
    glog::glog

    # ${dialect_libs}
    # ${conversion_libs}
    # ${extension_libs}
    # MLIRAnalysis
    # MLIRBuiltinToLLVMIRTranslation
    # MLIRCallInterfaces
    MLIRCastInterfaces
    # MLIRExecutionEngine
    MLIRFunctionInterfaces
    # MLIRIR
    # MLIRLLVMCommonConversion
    # MLIRLLVMToLLVMIRTranslation
    # MLIRMemRefDialect
    # MLIRParser
    # MLIRPass
    # MLIRSideEffectInterfaces
    # MLIRTargetLLVMIRExport
    # MLIRTransforms
)