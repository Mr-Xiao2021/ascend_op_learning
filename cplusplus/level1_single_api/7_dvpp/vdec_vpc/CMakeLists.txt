
# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(vdec_vpc_demo)

# Compile options
add_compile_options(-std=c++11)

set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall -g")

set(INC_PATH $ENV{DDK_PATH})
if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else ()
    message(STATUS "env INC_PATH: ${INC_PATH}")
endif()

set(LIB_PATH $ENV{NPU_HOST_LIB})
# Dynamic libraries in the stub directory can only be used for compilation
if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "/usr/local/Ascend/runtime/lib64/stub/")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else ()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif()

link_directories(
    ${LIB_PATH}
)

include_directories(
    ${INC_PATH}/runtime/include/acl/
    ${INC_PATH}/runtime/include/acl/dvpp
    ./src/
)

add_executable (vdec_vpc_demo
    ./src/demo.cpp
)

target_link_libraries(vdec_vpc_demo
    libacl_dvpp_mpi.so
    libascendcl.so
    pthread
    stdc++
)

