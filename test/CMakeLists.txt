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
    glog::glog
    # onnx_parser
)
# add_executable(test_parser test_parser.cpp)
# target_link_libraries(test_parser PRIVATE ${TESTLIB})

# add_executable(test_attribute test_attribute.cpp)
# target_link_libraries(test_attribute PRIVATE ${TESTLIB})

# add_executable(test_node test_node.cpp)
# target_link_libraries(test_node PRIVATE ${TESTLIB})

# add_executable(test_tensor test_tensor.cpp)
# target_link_libraries(test_tensor PRIVATE ${TESTLIB} ${PROTOBUF_LIBRARY_DEBUG} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# add_executable(test_onnx_parser test_onnx_parser.cpp)
# target_link_libraries(test_onnx_parser PRIVATE ${TESTLIB} ${PROTOBUF_LIBRARY_DEBUG} )

# add_executable(test_onnx_forward test_onnx_forward.cpp)
# target_link_libraries(test_onnx_forward PRIVATE ${TESTLIB} ${PROTOBUF_LIBRARY_DEBUG} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

add_executable(test_arma test_arma.cpp)
target_link_libraries(test_arma PRIVATE ${TESTLIB} ${PROTOBUF_LIBRARY_DEBUG} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

add_executable(test_ops test_ops.cpp)
target_link_libraries(test_ops PRIVATE ${TESTLIB} ${PROTOBUF_LIBRARY_DEBUG} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

add_executable(test_add test_add.cc)
target_link_libraries(test_add kernel_op_imp)