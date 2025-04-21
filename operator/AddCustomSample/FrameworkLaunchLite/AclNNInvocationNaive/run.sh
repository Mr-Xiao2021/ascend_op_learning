#!/bin/bash
SHORT=r:,v:,i:,
LONG=run-mode:,soc-version:,install-path:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

while :; do
    case "$1" in
    -r | --run-mode)
        RUN_MODE="$2"
        shift 2
        ;;
    -v | --soc-version)
        SOC_VERSION="$2"
        shift 2
        ;;
    -i | --install-path)
        ASCEND_INSTALL_PATH="$2"
        shift 2
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "[ERROR] Unexpected option: $1"
        break
        ;;
    esac
done

RUN_MODE_LIST="sim npu"
if [[ " $RUN_MODE_LIST " != *" $RUN_MODE "* ]]; then
    echo "ERROR: RUN_MODE error, This sample only support sim or npu!"
    exit -1
fi

if [ "${RUN_MODE}" = "npu" ] && [ "$SOC_VERSION" ]; then
    echo "ERROR: can not specify SOC_VERSION when running on npu!"
    exit -1
fi

VERSION_LIST="Ascend910A Ascend910B Ascend310B1 Ascend310B2 Ascend310B3 Ascend310B4 Ascend310P1 Ascend310P3 Ascend910B1 Ascend910B2 Ascend910B3 Ascend910B4"
if [ "${RUN_MODE}" = "sim" ] && [[ " $VERSION_LIST " != *" $SOC_VERSION "* ]]; then
    echo "ERROR: SOC_VERSION should be in [$VERSION_LIST]"
    exit -1
fi

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
source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash
export DDK_PATH=${_ASCEND_INSTALL_PATH}
export NPU_HOST_LIB=${_ASCEND_INSTALL_PATH}/lib64

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}
if [ "${RUN_MODE}" = "sim" ]; then
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
fi

set -e
rm -rf build
mkdir -p build
cmake -B build
cmake --build build -j
(
    cd build
    if [[ "$RUN_WITH_TOOLCHAIN" -eq 1 ]]; then
        if [ "${RUN_MODE}" = "npu" ]; then
            msprof op --application=./execute_add_op
        elif [ "${RUN_MODE}" = "sim" ]; then
            msprof op simulator --application=./execute_add_op
        fi
    else
        ./execute_add_op
    fi
)