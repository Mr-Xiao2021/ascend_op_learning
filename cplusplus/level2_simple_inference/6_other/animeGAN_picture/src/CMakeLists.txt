# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(animeGAN_picture)

# Compile options
add_compile_options(-std=c++11)

add_definitions(-DENABLE_DVPP_INTERFACE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../../../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

if (NOT DEFINED ENV{INSTALL_DIR})
    message(FATAL_ERROR "Not Defined INSTALL_DIR")
endif()

if (NOT DEFINED ENV{THIRDPART_PATH})
    message(FATAL_ERROR "Not Defined THIRDPART_PATH")
endif()

if (NOT DEFINED ENV{CPU_ARCH})
    message(FATAL_ERROR "Not Defined CPU_ARCH")
endif()

add_definitions(-DENABLE_DVPP_INTERFACE)
list(APPEND COMMON_DEPEND_LIB acllite avcodec avformat avdevice avutil swresample avfilter swscale)
if ($ENV{CPU_ARCH} MATCHES "aarch64")
    if(EXISTS "$ENV{INSTALL_DIR}/driver/libmedia_mini.so")
        list(APPEND COMMON_DEPEND_LIB media_mini ascend_hal c_sec mmpa slog)
        add_definitions(-DENABLE_BOARD_CAMARE)
        message(STATUS "arch: arm")
    endif()
endif()

# Header path
include_directories(
    $ENV{THIRDPART_PATH}/include/
    $ENV{THIRDPART_PATH}/include/acllite
    $ENV{THIRDPART_PATH}/include/opencv4
    $ENV{INSTALL_DIR}/driver    
    $ENV{INSTALL_DIR}/runtime/include/
    ../inc/
)

if(target STREQUAL "Simulator_Function")
    add_compile_options(-DFUNC_SIM)
endif()

# add host lib path
link_directories(
    $ENV{THIRDPART_PATH}/lib/
    $ENV{INSTALL_DIR}/driver
    $ENV{INSTALL_DIR}/runtime/lib64/stub
)

add_executable(main
        # utils.cpp
        object_detect.cpp
        main.cpp)

if(target STREQUAL "Simulator_Function")
    target_link_libraries(main funcsim)
else()
    target_link_libraries(main ascendcl acl_dvpp stdc++ pthread ${COMMON_DEPEND_LIB} opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d opencv_features2d opencv_videoio protobuf presenteragent dl rt)
endif()

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
