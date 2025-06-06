# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(ACL_RESNET50)

# Compile options
add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../../../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

set(INC_PATH $ENV{DDK_PATH})

if (NOT DEFINED ENV{DDK_PATH})
    if (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Windows")
        set(INC_PATH "C:/Program Files/HuaWei/Ascend")
    else ()
        set(INC_PATH "/usr/local/Ascend")
    endif ()
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else ()
    message(STATUS "env INC_PATH: ${INC_PATH}")
endif ()

set(LIB_PATH $ENV{NPU_HOST_LIB})

# Dynamic libraries in the stub directory can only be used for compilation
if (NOT DEFINED ENV{NPU_HOST_LIB})
    if (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Windows")
        set(LIB_PATH "C:/Program Files/HuaWei/Ascend/Acllib/lib64")
    else ()
        set(LIB_PATH "/usr/local/Ascend/runtime/lib64/stub/")
    endif ()
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else ()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif ()

# Header path
include_directories(
    ${INC_PATH}/runtime/include/

)

if(target STREQUAL "Simulator_Function")
    add_compile_options(-DFUNC_SIM)
endif()

# add host lib path
link_directories(
    ${LIB_PATH}
)

add_executable(main
        main.cpp)

if (target STREQUAL "Simulator_Function")
    target_link_libraries(main funcsim)
else ()
    if (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Windows")
        target_link_libraries(main
            libascendcl)
    else ()
        target_link_libraries(main
            ascendcl stdc++)
    endif ()
endif ()

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})