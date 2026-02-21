#!/bin/bash
# build and run script for the lock free histogram benchmark
set -e

nvcc -O3 -arch=native -allow-unsupported-compiler main.cu -o hist_bench
./hist_bench
