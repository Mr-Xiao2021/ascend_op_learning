#!/bin/bash
if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi
source $_ASCEND_INSTALL_PATH/bin/setenv.bash
export DDK_PATH=$_ASCEND_INSTALL_PATH
export NPU_HOST_LIB=$_ASCEND_INSTALL_PATH/lib64

set -e
rm -rf build
mkdir -p build
cmake -B build
cmake --build build -j
(
    cd build
    ./execute_add_op
)
