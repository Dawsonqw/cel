set(LIBS
    CelBasic
    CelSupport
    glog::glog
    MLIRAnalysis
    MLIRBuiltinToLLVMIRTranslation
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRExecutionEngine
    MLIRFunctionInterfaces
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMToLLVMIRTranslation
    MLIRMemRefDialect
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRTargetLLVMIRExport
    MLIRTransforms
)

set(DEPEND
    cel_dialect_gen
)

# add_executable(test_opt test_opt.cpp)
# add_dependencies(test_opt ${DEPEND})
# target_link_libraries(test_opt PRIVATE ${LIBS})


# add_executable(test_parser test_parser.cpp)

# add_executable(test_val test_val.cpp)


# unit tests
set(
    TESTLIB
    ${GTEST_BOTH_LIBRARIES}
    onnx_parser
)
# add_executable(test_attribute test_attribute.cpp)
# target_link_libraries(test_attribute PRIVATE ${TESTLIB})

# add_executable(test_node test_node.cpp)
# target_link_libraries(test_node PRIVATE ${TESTLIB})

add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE ${TESTLIB})
