#!/bin/bash

ScriptPath="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"
 
echo "[INFO] The sample starts to run"
cd ${ScriptPath}/../src
python3 sampleResnetRtsp.py 
if [ $? -ne 0 ];then
    echo "[INFO] The program runs failed"
else
    echo "[INFO] The program runs successfully"
fi
