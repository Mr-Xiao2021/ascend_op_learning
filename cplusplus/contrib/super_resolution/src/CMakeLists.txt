# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(sample-superresolution)

# Compile options
add_compile_options(-std=c++11)

add_definitions(-DENABLE_DVPP_INTERFACE)

# Specify target generation path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../../../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

if (NOT DEFINED ENV{INSTALL_DIR})
    MESSAGE(FATAL_ERROR "Not Defined INSTALL_DIR")
endif()

if (NOT DEFINED ENV{THIRDPART_PATH})
    MESSAGE(FATAL_ERROR "Not Defined THIRDPART_PATH")
endif()

if (NOT DEFINED ENV{CPU_ARCH})
    MESSAGE(FATAL_ERROR "Not Defined CPU_ARCH")
endif()

add_definitions(-DENABLE_DVPP_INTERFACE)
list(APPEND COMMON_DEPEND_LIB acllite avcodec avformat avdevice avutil swresample avfilter swscale)
if ($ENV{CPU_ARCH} MATCHES "aarch64")
    if(EXISTS "$ENV{INSTALL_DIR}/driver/libmedia_mini.so")
        list(APPEND COMMON_DEPEND_LIB media_mini ascend_hal c_sec mmpa)
        add_definitions(-DENABLE_BOARD_CAMARE)
        message(STATUS "arch: arm")
    endif()
endif()

# Header path
include_directories(
    ../inc/
    $ENV{INSTALL_DIR}/runtime/include/
    $ENV{THIRDPART_PATH}/include
)


# add host lib path
link_directories(
    $ENV{INSTALL_DIR}/runtime/lib64/stub
    $ENV{THIRDPART_PATH}/lib
    $ENV{INSTALL_DIR}/driver
)

add_executable(main
        utils.cpp
        super_resolution_process.cpp
        model_process.cpp
        main.cpp)

target_link_libraries(main
        ascendcl acl_dvpp stdc++ ${COMMON_DEPEND_LIB} opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d opencv_features2d opencv_videoio)

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
