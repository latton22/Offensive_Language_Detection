#!/bin/tcsh

mkdir -p ../out/summary

python3 ../tool/summary_result.py ../config/config.py ../out
