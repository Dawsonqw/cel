file(
    GLOB Compute_SOURCES
    "*.cc"
)

add_library(
    Compute
    STATIC
    ${Compute_SOURCES}
)