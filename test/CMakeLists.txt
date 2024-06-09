set(LIBS
    CelBasic
    CelSupport
    glog::glog
)

set(DEPEND
    cel_dialect_gen
)

add_executable(test_opt test_opt.cpp)
add_dependencies(test_opt ${DEPEND})
target_link_libraries(test_opt PRIVATE ${LIBS})