ifeq (${ASCEND_INSTALL_PATH},)
    ASCEND_PATH := /usr/local/Ascend
else
    ASCEND_PATH := ${ASCEND_INSTALL_PATH}
endif


LOCAL_DIR  := ./
ATC_INCLUDE_DIR := $(ASCEND_PATH)/compiler/include
OPP_INCLUDE_DIR := $(ASCEND_PATH)/opp/built-in/op_proto/inc

LOCAL_MODULE_NAME := ir_build
CC := g++
CFLAGS := -std=c++11 -g -Wall -D_GLIBCXX_USE_CXX11_ABI=0
SRCS := $(wildcard $(LOCAL_DIR)/src/main.cpp)

INCLUDES := -I $(ASCEND_PATH)/opp/built-in/op_proto/inc \
            -I $(ATC_INCLUDE_DIR)/graph \
            -I $(ATC_INCLUDE_DIR)/ge \
            -I $(ATC_INCLUDE_DIR)/parser \
            -I $(ASCEND_PATH)/compiler/include \

LIBS := -L ${ASCEND_PATH}/compiler/lib64/stub \
    -lgraph \
    -lge_compiler \
    -lfmk_parser \

ir_build:
	mkdir -p out
	$(CC) $(SRCS) $(INCLUDES) $(LIBS) $(CFLAGS) -o ./out/$(LOCAL_MODULE_NAME)

clean:
