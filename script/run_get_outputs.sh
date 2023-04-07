#!/bin/tcsh

set gpu_id = 0

mkdir -p ../out

python3 ../tool/get_outputs.py ../config/config.py $gpu_id
