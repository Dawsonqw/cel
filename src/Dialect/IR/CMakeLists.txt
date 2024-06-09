file(GLOB _sources
          CelOps.cpp
    )

add_mlir_library(CelBasic
    ${_sources}
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include

    DEPENDS
    cel_dialect_gen

    LINK_LIBS PRIVATE
    MLIR
)