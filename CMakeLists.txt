cmake_minimum_required(VERSION 3.20)
project(cel LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50")
set(CMAKE_CUDA_STANDARD 11)
set(CUDA_ARCHITECTURES 50)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package (glog 0.7.0 REQUIRED)
find_package(GTest)
find_package(Protobuf REQUIRED)
find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
# find_package(OpenBLAS REQUIRED)

add_definitions(-DGLOG_USE_GLOG_EXPORT)
add_definitions(-DARMA_DONT_USE_WRAPPER)
 
if (Protobuf_FOUND)
    MESSAGE(STATUS "PROTOBUF FOUNED, ${PROTOBUF_INCLUDE_DIR}")
    INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})
    link_directories(${PROTOBUF_LIBRARIES})
else (Protobuf_FOUND)
    MESSAGE(FATAL_ERROR "PROTOBUF NOT FOUNED")
endif (Protobuf_FOUND)


if(BLAS_FOUND)
    MESSAGE(STATUS "BLAS FOUNED, ${BLAS_INCLUDE_DIR}")
    INCLUDE_DIRECTORIES(${BLAS_INCLUDE_DIR})
    link_directories(${BLAS_LIBRARIES})
else (BLAS_FOUND)
    MESSAGE(FATAL_ERROR "BLAS NOT FOUNED")
endif(BLAS_FOUND)

if(LAPACK_FOUND)
    MESSAGE(STATUS "LAPACK FOUNED, ${LAPACK_INCLUDE_DIR}")
    INCLUDE_DIRECTORIES(${LAPACK_INCLUDE_DIR})
    link_directories(${LAPACK_LIBRARIES})
else (LAPACK_FOUND)
    MESSAGE(FATAL_ERROR "LAPACK NOT FOUNED")
endif(LAPACK_FOUND)


message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using glogConfig.cmake in: ${glog_DIR}")
message(STATUS "Using GTestConfig.cmake in: ${GTest_DIR}")
message(STATUS "Using glogConfig.cmake in: ${glog_DIR}")
# message(STATUS "Using OpenBLASConfig.cmake in: ${OpenBLAS_DIR}")


list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddMLIRPython)
include(MLIRDetectPythonEnv)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR}/cuda)
include_directories(/usr/local/cuda/include)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

add_subdirectory(cuda)
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)