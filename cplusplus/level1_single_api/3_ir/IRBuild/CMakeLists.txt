cmake_minimum_required(VERSION 3.5.1)
project(GE_IR[CXX])

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

if (DEFINED ENV{ASCEND_INSTALL_PATH})
    set(ASCEND_PATH $ENV{ASCEND_INSTALL_PATH})
else ()
    set(ASCEND_PATH /usr/local/Ascend)
endif()

set(ATC_INCLUDE_DIR ${ASCEND_PATH}/compiler/include)

set(SRC_LIST
    "src/main.cpp"
)

set(common_compile_options
    -std=c++11
    -g
    -Wall    
)

set(common_compile_definitions
    _GLIBCXX_USE_CXX11_ABI=0	
)

############ ir_build ############
add_executable(ir_build
    ${SRC_LIST}
)

target_compile_options(ir_build PRIVATE
    ${common_compile_options}
)

target_compile_definitions(ir_build PRIVATE
    ${common_compile_definitions}	
)

target_include_directories(ir_build PRIVATE
    ${ASCEND_PATH}/opp/built-in/op_proto/inc
    ${ATC_INCLUDE_DIR}/graph
    ${ATC_INCLUDE_DIR}/ge
    ${ATC_INCLUDE_DIR}/parser    
    ${ATC_INCLUDE_DIR}
)

target_link_directories(ir_build PRIVATE
    ${ASCEND_PATH}/compiler/lib64/stub	
)

target_link_libraries(ir_build PRIVATE
    -Wl,--no-as-needed
    graph
    ge_compiler
    fmk_parser
    -Wl,--as-needed    
)

############ install ############
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR})

install(TARGETS ir_build OPTIONAL
    RUNTIME DESTINATION output	
)
