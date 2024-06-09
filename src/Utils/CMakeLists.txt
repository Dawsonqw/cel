file(GLOB _sources
          common.cpp
    )

add_mlir_library(CelSupport
    ${_sources}
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include

    DEPENDS

    LINK_LIBS PRIVATE
    glog::glog
)