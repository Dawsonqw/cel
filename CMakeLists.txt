cmake_minimum_required(VERSION 3.20)
project(cel)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
include(TableGen)
include(AddLLVM)
include(AddMLIR)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

add_subdirectory(${PROJECT_SOURCE_DIR}/include)